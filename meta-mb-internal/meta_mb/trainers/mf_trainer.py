import torch
import numpy as np
from meta_mb.logger import logger
from meta_mb.samplers.utils import rollout

from babyai.rl.utils.dictlist import merge_dictlists
from babyai.utils.buffer import Buffer, trim_batch
# from scripts.test_generalization import eval_policy, test_success
import os.path as osp
import joblib
import time
import psutil
import os
import copy
import random
import pathlib

from envs.d4rl_envs import AntEnv, PointMassEnv


class Trainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) : 
        sample_processor (SampleProcessor) : 
        baseline (Baseline) : 
        policy (Policy) : 
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
    """

    def __init__(
            self,
            args,
            algo,
            algo_dagger,
            policy,
            reward,
            env,
            sampler,
            sample_processor,
            start_itr=0,
            buffer_name="",
            exp_name="",
            curriculum_step=0,
            il_trainer=None,
            il_trainer_rew=None,
            reward_predictor=None,
            rp_trainer=None,
            is_debug=False,
            eval_every=100,
            save_every=100,
            log_every=10,
            save_videos_every=1000,
            log_and_save=True,
            teacher_schedule=lambda a, b: ({}, {}),
            obs_preprocessor=None,
            log_dict={},
            eval_heldout=True,
            augmenter=None,
            log_fn=lambda w, x, y, z: None,
    ):
        self.args = args
        self.algo = algo
        self.algo_dagger = algo_dagger
        self.policy_dict = policy
        self.rew_dict = reward
        self.env = copy.deepcopy(env)
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.start_itr = start_itr
        self.buffer_name = buffer_name
        self.exp_name = exp_name
        self.curriculum_step = curriculum_step
        self.il_trainer = il_trainer
        self.il_trainer_rew = il_trainer_rew
        self.reward_predictor = reward_predictor
        self.rp_trainer = rp_trainer
        self.is_debug = is_debug
        self.eval_every = eval_every
        self.save_every = save_every
        self.log_every = log_every
        self.save_videos_every = save_videos_every
        self.log_and_save = log_and_save
        self.train_with_teacher = args.feedback_type is not None
        self.introduced_teachers = log_dict.get('introduced_teachers', set())
        self.num_feedback_advice = log_dict.get('num_feedback_advice', 0)
        self.num_feedback_reward = log_dict.get('num_feedback_reward', 0)
        self.total_distillation_frames = log_dict.get('total_distillation_frames', 0)
        self.itrs_on_level = log_dict.get('itrs_on_level', 0)
        # Dict saying which teacher types the agent has access to for training.
        self.teacher_schedule = teacher_schedule
        # Dict specifying no teacher provided.
        self.no_teacher_dict, _ = teacher_schedule(-1, -1, -1)
        self.gave_feedback = log_dict.get('gave_feedback', {k: 0 for k in self.no_teacher_dict.keys()})
        self.followed_feedback = log_dict.get('followed_feedback', {k: 0 for k in self.no_teacher_dict.keys()})
        self.obs_preprocessor = obs_preprocessor
        self.next_train_itr = log_dict.get('next_train_itr', start_itr)
        self.num_train_skip_itrs = log_dict.get('num_train_skip_itrs', 5)
        self.eval_heldout = eval_heldout
        self.augmenter = augmenter
        self.log_fn = log_fn
        self.advancement_count_threshold = getattr(args, 'advancement_count', 1)
        self.advancement_count = 0
        self.success_dict = {k: 0 for k in self.no_teacher_dict.keys()}
        self.success_dict['none'] = 0
        self.best_train_perf = (0, 0, float('inf'))  # success, accuracy, loss
        self.itrs_since_best = 0

    def check_advance_curriculum_train(self, episode_logs, data):
        if False:  # data.obs[0]['gave_PreActionAdvice']:
            acc_threshold = .98
        else:
            acc_threshold = self.args.accuracy_threshold_rl
        if episode_logs is None:
            return True, -1, -1
        if self.args.discrete:
            avg_accuracy = torch.eq(data.action_probs.argmax(dim=1),
                                    data.teacher_action).float().mean().item()
        else:
            avg_accuracy = 0
        avg_success = np.mean(episode_logs["success_per_episode"])
        should_advance_curriculum = (avg_success >= self.args.success_threshold_rl) \
                                    and (avg_accuracy >= acc_threshold)
        best_success, best_accuracy, best_loss = self.best_train_perf
        if avg_success > best_success or (avg_success == best_success and avg_accuracy > best_accuracy):
            if self.args.early_stop_metric == 'success':
                self.itrs_since_best = 0
            self.best_train_perf = (avg_success, avg_accuracy, best_loss)
        else:
            if self.args.early_stop_metric == 'success':
                self.itrs_since_best += 1
        return should_advance_curriculum, avg_success, avg_accuracy

    def check_advance_curriculum_rollout(self, data, use_teacher):
        num_total_episodes = data['dones'].sum()
        num_successes = data['env_infos']['success'].sum()
        avg_success = num_successes / num_total_episodes
        # Episode length contains the timestep, starting at 1.  Padding values are 0.
        pad_steps = (data['env_infos']['episode_length'] == 0).sum()
        total_steps = (np.prod(data['actions'].shape) - pad_steps)
        if self.args.discrete:
            assert data['actions'].shape == data['env_infos']['teacher_action'].shape
            correct_actions = (data['actions'] == data['env_infos']['teacher_action']).sum() - pad_steps
            avg_accuracy = correct_actions / total_steps
        else:
            avg_accuracy = 0
        avg_reward = data['rewards'].sum() / total_steps
        # We take the max since runs which end early will be 0-padded
        if use_teacher:
            success_threshold = self.args.success_threshold_rollout_teacher
            accuracy_threshold = self.args.accuracy_threshold_rollout_teacher
        else:
            success_threshold = self.args.success_threshold_rollout_no_teacher
            accuracy_threshold = self.args.accuracy_threshold_rollout_no_teacher
        should_advance_curriculum = avg_success >= success_threshold and avg_accuracy >= accuracy_threshold
        return should_advance_curriculum, avg_success, avg_accuracy, avg_reward

    def train(self):
        """
        Trains policy on env using algo
        """
        start_time = time.time()
        rollout_time = 0
        saving_time = 0

        if not getattr(self.args, 'no_buffer', False):
            buffer = Buffer(self.buffer_name, self.args.buffer_capacity, self.args.prob_current, val_prob=.1,
                            augmenter=self.augmenter, successful_only=self.args.distill_successful_only,
                            sample_frac=self.args.sample_frac, sample_strategy=self.args.sample_strategy)
            self.buffer = buffer
            if self.args.use_dagger:
                dagger_buffer = Buffer(self.buffer_name, self.args.buffer_capacity, self.args.prob_current, val_prob=.1,
                                       buffer_name='dagger_buffer', successful_only=self.args.distill_successful_only)
            else:
                dagger_buffer = None
        else:
            buffer = None

        itr_start_time = time.time()

        all_time_training = 0
        all_time_collection = 0
        all_rp_splice_time = 0
        all_time_rp_train = 0
        all_run_policy_time = 0
        all_distill_time = 0
        all_rollout_time = 0
        all_saving_time = 0
        all_unaccounted_time = 0
        last_success = 0
        last_accuracy = 0

        for itr in range(self.start_itr, self.args.n_itr):

            if False:
                params = self.get_itr_snapshot(itr)
                step = self.curriculum_step
                logger.save_itr_params(itr, step, params)
                assert False
            if itr % self.log_every == 0:
                if self.args.feedback_from_buffer:
                    num_feedback = self.buffer.num_feedback
                else:
                    num_feedback = self.num_feedback_advice + self.num_feedback_reward
                self.log_fn(self.policy_dict, logger, itr, num_feedback)

            teacher_train_dict, teacher_distill_dict = self.teacher_schedule(self.curriculum_step,
                                                                             last_success,
                                                                             last_accuracy)
            collection_dict = {k: teacher_train_dict[k] or teacher_distill_dict[k] for k in teacher_train_dict.keys()}
            if len(teacher_train_dict) > 0:
                last_teacher = list(teacher_train_dict.keys())[-1]
            else:
                last_teacher = 'none'
            for teacher_name, teacher_present in teacher_train_dict.items():
                if teacher_present:
                    self.introduced_teachers.add(teacher_name)
            # Not using any teacher
            if np.sum([int(v) for v in teacher_train_dict.values()]) == 0:
                self.introduced_teachers.add('none')
            if self.il_trainer is not None:
                teachers = ['all_teachers', 'all_but_none', 'powerset', 'single_teachers', 'single_teachers_none']
                if self.args.distillation_strategy in teachers:
                    for teacher_name, teacher_present in teacher_distill_dict.items():
                        if teacher_present:
                            self.introduced_teachers.add(teacher_name)
                if self.args.distillation_strategy in ['no_teachers', 'powerset', 'single_teachers_none']:
                    self.introduced_teachers.add('none')

            logger.logkv("ItrsOnLevel", self.itrs_on_level)
            self.itrs_on_level += 1

            # If we're distilling, don't train the first time on the level in case we can zero-shot it
            skip_training_rl = self.args.reward_when_necessary and not self.next_train_itr == itr

            logger.log("\n ---------------- Iteration %d ----------------" % itr)
            logger.log("Sampling set of tasks/goals for this meta-batch...")

            """ -------------------- Sampling --------------------------"""

            logger.log("Obtaining samples...")
            time_env_sampling_start = time.time()
            should_collect = (not self.args.no_collect) and (
                    (not skip_training_rl) or self.args.self_distill)
            should_train_rl = not (self.args.no_collect or self.args.no_train_rl or skip_training_rl)
            #should_collect = should_collect and ((not self.curriculum_step in self.buffer.counts_train) or self.buffer.counts_train[self.curriculum_step] < self.buffer.train_buffer_capacity)
            if should_collect:
                # Collect if we are distilling OR if we're not skipping
                input_dict = teacher_distill_dict if (self.args.relabel_goal
                                                      or self.args.relabel
                                                      or (self.args.half_relabel and itr % 2 == 0)
                                                      ) else teacher_train_dict
                samples_data, episode_logs = self.algo.collect_experiences(input_dict,
                                                                           collect_with_oracle=self.args.collect_with_oracle,
                                                                           collect_reward=should_train_rl,
                                                                           train=should_train_rl,
                                                                           collection_dict=collection_dict)
                if self.args.relabel_goal:
                    print("relabeling!")
                    samples_data = self.relabel_goal(samples_data)
                raw_samples_data = copy.deepcopy(samples_data)
                try:
                    counts_train = buffer.counts_train[self.curriculum_step]
                except:
                    counts_train = 0
                logger.logkv("BufferSize", counts_train)
                if self.args.end_on_full_buffer and \
                        (buffer.counts_train[self.curriculum_step] == buffer.train_buffer_capacity):
                    print("ALL DONE!")
                    return
            else:
                print("Not collecting")
                episode_logs = None
                raw_samples_data = None
                samples_data = None

            """ -------------------- Training --------------------------"""

            time_collection = time.time() - time_env_sampling_start
            time_training_start = time.time()
            if should_train_rl:
                early_entropy_coef = self.args.early_entropy_coef if self.itrs_on_level < 10 else None
                summary_logs = self.algo.optimize_policy(samples_data, teacher_dict=teacher_train_dict,
                                                         entropy_coef=early_entropy_coef)
            else:
                summary_logs = None
            time_training = time.time() - time_training_start
            self._log(episode_logs, summary_logs, samples_data, tag="Train")
            logger.logkv('Curriculum Step', self.curriculum_step)
            advance_curriculum, avg_success, avg_accuracy = self.check_advance_curriculum_train(episode_logs,
                                                                                                raw_samples_data)
            if self.args.no_train_rl or skip_training_rl:
                advance_curriculum = True
            else:
                # Decide whether to train RL next itr
                if advance_curriculum:
                    self.next_train_itr = itr + self.num_train_skip_itrs
                    self.num_train_skip_itrs += 5
                else:
                    self.next_train_itr = itr + 1
                    self.num_train_skip_itrs = 5
            should_store_data = raw_samples_data is not None and (
                    self.args.collect_before_threshold or advance_curriculum) and not getattr(self.args, 'no_buffer', False)
            if self.args.yes_distill:
                should_store_data = raw_samples_data is not None
            if should_store_data:
                buffer.add_batch(raw_samples_data, self.curriculum_step)

                if self.args.use_dagger:
                    for i in range(1):
                        dagger_samples_data, _ = self.algo_dagger.collect_experiences(teacher_train_dict,
                                                                                      use_dagger=True,
                                                                                      dagger_dict={
                                                                                          k: k == 'CartesianCorrections'
                                                                                          for k in
                                                                                          self.no_teacher_dict.keys()})
                        dagger_buffer.add_batch(dagger_samples_data, self.curriculum_step)
                else:
                    dagger_samples_data = None

            logger.logkv('Train/Advance', int(advance_curriculum))

            # """ ------------------ Reward Predictor Splicing ---------------------"""
            rp_start_time = time.time()
            # samples_data = self.use_reward_predictor(samples_data)  # TODO: update
            rp_splice_time = time.time() - rp_start_time

            """ ------------------ Policy Update ---------------------"""

            logger.log("Optimizing policy...")
            # # This needs to take all samples_data so that it can construct graph for meta-optimization.
            time_rp_train_start = time.time()
            # self.train_rp(samples_data)
            time_rp_train = time.time() - time_rp_train_start

            """ ------------------ Reward Training ---------------------"""
            if self.args.should_train_rew:
                logger.log("Training reward ...")
                for dist_i in range(self.args.distillation_steps):
                    sampled_batch = buffer.sample(total_num_samples=self.args.batch_size, split='train')
                    rew_log = self.distill(sampled_batch,
                                               is_training=True,
                                               teachers_dict=teacher_distill_dict,
                                               relabel=False,#self.args.relabel_rew,
                                               relabel_dict=teacher_train_dict, distill_to_none=True, train_reward=True)
                for key_set, log_dict in rew_log.items():
                    key_set = '_'.join(key_set)
                    for k, v in log_dict.items():
                        logger.logkv(f"Reward/{key_set}{k}_Train", v)
                sampled_val_batch = buffer.sample(total_num_samples=self.args.batch_size,
                                                  split='val')
                rew_log_val = self.distill(sampled_val_batch,
                                               is_training=False,
                                               #source='teacher',
                                               teachers_dict=teacher_distill_dict,
                                               relabel=False,#self.args.relabel_rew,
                                               relabel_dict=teacher_train_dict,
                                               train_reward=True)

                for key_set, log_dict in rew_log_val.items():
                    key_set = '_'.join(key_set)
                    for k, v in log_dict.items():
                        logger.logkv(f"Reward/{key_set}{k}_Val", v)

            """ ------------------ Distillation ---------------------"""
            should_distill = self.args.self_distill and advance_curriculum and \
                             self.itrs_on_level >= self.args.min_itr_steps_distill
            if self.args.yes_distill:
                should_distill = self.itrs_on_level >= self.args.min_itr_steps_distill
            if self.args.no_distill:
                should_distill = False
            if buffer is not None and sum(list(buffer.counts_train.values())) == 0:
                should_distill = False
            if should_distill:
                logger.log("Distilling ...")
                time_distill_start = time.time()
                time_sampling_from_buffer = 0
                time_train_distill = 0
                time_val_distill = 0
                for dist_i in range(self.args.distillation_steps):
                    sample_start = time.time()
                    sampled_batch = buffer.sample(total_num_samples=self.args.batch_size, split='train')
                    time_sampling_from_buffer += (time.time() - sample_start)
                    sample_start = time.time()
                    self.total_distillation_frames += len(sampled_batch)
                    distill_log = self.distill(sampled_batch,
                                               is_training=True,
                                               teachers_dict=teacher_distill_dict,
                                               relabel=self.args.relabel or (self.args.half_relabel and itr % 2 == 0),
                                               relabel_dict=teacher_train_dict, distill_to_none=True)  # dist_i < 5)
                    time_train_distill += (time.time() - sample_start)
                    if self.args.use_dagger:
                        sampled_dagger_batch = dagger_buffer.sample(total_num_samples=self.args.batch_size,
                                                                    split='train')
                        self.total_distillation_frames += len(sampled_dagger_batch)
                        dagger_distill_log = self.distill(sampled_dagger_batch,
                                                          is_training=True,
                                                          source='teacher',
                                                          teachers_dict=teacher_distill_dict,
                                                          relabel=self.args.relabel or (self.args.half_relabel and itr % 2 == 0),
                                                          relabel_dict=teacher_train_dict)
                        distill_log = dagger_distill_log
                        for key_set, log_dict in dagger_distill_log.items():
                            key_set = '_'.join(key_set)
                            for k, v in log_dict.items():
                                logger.logkv(f'Distill/DAgger_{key_set}{k}_Train', v)

                for key_set, log_dict in distill_log.items():
                    key_set = '_'.join(key_set)
                    for k, v in log_dict.items():
                        logger.logkv(f"Distill/{key_set}{k}_Train", v)
                sample_start = time.time()
                time_sampling_from_buffer += (time.time() - sample_start)
                sample_start = time.time()
                sampled_val_batch = buffer.sample(total_num_samples=self.args.batch_size,
                                                  split='val')
                distill_log_val = self.distill(sampled_val_batch,
                                               is_training=False,
                                               #source='teacher',
                                               teachers_dict=teacher_distill_dict,
                                               relabel=self.args.relabel or (self.args.half_relabel and itr % 2 == 0),
                                               relabel_dict=teacher_train_dict)

                time_val_distill += (time.time() - sample_start)
                best_success, best_accuracy, best_loss = self.best_train_perf
                val_loss_key = () if () in distill_log_val.keys() else list(distill_log_val.keys())[0]
                val_loss = distill_log_val[val_loss_key]['TotalLoss']
                if val_loss < best_loss:
                    self.best_train_perf = (best_success, best_accuracy, val_loss)
                    if self.args.early_stop_metric == 'val_loss':
                        self.itrs_since_best = 0
                else:
                    if self.args.early_stop_metric == 'val_loss':
                        self.itrs_since_best += 1
                for key_set, log_dict in distill_log_val.items():
                    key_set = '_'.join(key_set)
                    for k, v in log_dict.items():
                        logger.logkv(f"Distill/{key_set}{k}_Val", v)
                distill_time = time.time() - time_distill_start
                advance_curriculum = True
                for teacher_key_set in distill_log_val.keys():
                    acc = distill_log_val[teacher_key_set]['Accuracy']
                    if teacher_key_set == ():
                        advance_teacher = acc >= self.args.accuracy_threshold_distill_no_teacher
                    else:
                        advance_teacher = acc >= self.args.accuracy_threshold_distill_teacher
                    key_set_name = '_'.join(list(teacher_key_set))
                    logger.logkv(f'Distill/Advance_{key_set_name}', int(advance_teacher))
                    advance_curriculum = advance_curriculum and advance_teacher
                logger.logkv('Distill/Advance_Overall', int(advance_curriculum))
                logger.logkv('Distill/TotalFrames', self.total_distillation_frames)

            else:
                distill_time = 0

            """ ------------------ Policy rollouts ---------------------"""
            should_policy_rollout = (itr % self.eval_every == 0) or (
                    itr == self.args.n_itr - 1)
            if self.args.yes_rollouts:
                should_policy_rollout = True
            if self.args.no_rollouts:
                should_policy_rollout = False
            if should_policy_rollout:
                run_policy_start = time.time()
                train_advance_curriculum = advance_curriculum
                with torch.no_grad():
                    logger.log("Running model with each teacher")
                    # Take distillation dict, keep the last teacher
                    for teacher in self.introduced_teachers:
                        advance_curriculum_teacher, success, accuracy = self.run_supervised(
                            self.policy_dict[teacher], {k: k == teacher for k in teacher_train_dict.keys()},
                            f"Rollout/",
                            show_instrs=True if teacher == 'none' else not self.args.rollout_without_instrs)
                        past_success = self.success_dict[teacher]
                        self.success_dict[teacher] = success * self.args.swap_factor + past_success * (
                                1 - self.args.swap_factor)
                        advance_curriculum_teacher = advance_curriculum_teacher and self.success_dict[teacher]
                        if teacher == last_teacher:
                            last_success = (
                                        success * self.args.swap_factor + last_success * (1 - self.args.swap_factor))
                            last_accuracy = (
                                        accuracy * self.args.swap_factor + last_accuracy * (1 - self.args.swap_factor))
                        advance_curriculum = advance_curriculum and advance_curriculum_teacher
                    advance_curriculum = advance_curriculum and train_advance_curriculum
                    print("Advancing curriculum???", advance_curriculum)

                    logger.logkv('Advance', int(advance_curriculum))
                run_policy_time = time.time() - run_policy_start
            else:
                advance_curriculum = False
                run_policy_time = 0

            logger.logkv('Rollout/last_success', last_success)
            logger.logkv('Rollout/last_accuracy', last_accuracy)

            """ ------------------- Logging Stuff --------------------------"""
            logger.logkv('Itr', itr)
            logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)
            logger.logkv('Train/SkipTrainRL', int(skip_training_rl))

            time_total = time.time() - start_time
            time_itr = time.time() - itr_start_time
            itr_start_time = time.time()
            logger.logkv('Time/Total', time_total)
            logger.logkv('Time/Itr', time_itr)

            try:
                logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.train_levels))
            except:
                print("no curriculum")

            process = psutil.Process(os.getpid())
            memory_use = process.memory_info().rss / float(2 ** 20)
            logger.logkv('Memory MiB', memory_use)

            logger.log(self.exp_name)

            logger.logkv('Time/Training', time_training)
            logger.logkv('Time/Collection', time_collection)
            logger.logkv('Time/RPUse', rp_splice_time)
            logger.logkv('Time/RPTrain', time_rp_train)
            logger.logkv('Time/RunwTeacher', run_policy_time)
            logger.logkv('Time/Distillation', distill_time)
            logger.logkv('Time/VidRollout', rollout_time)
            logger.logkv('Time/Saving', saving_time)
            time_unaccounted = time_itr - time_training - time_collection - \
                               rp_splice_time - time_rp_train - run_policy_time - distill_time - rollout_time - saving_time
            logger.logkv('Time/Unaccounted', time_unaccounted)

            all_time_training += time_training
            all_time_collection += time_collection
            all_rp_splice_time += rp_splice_time
            all_time_rp_train += time_rp_train
            all_run_policy_time += run_policy_time
            all_distill_time += distill_time
            all_rollout_time += rollout_time
            all_saving_time += saving_time
            all_unaccounted_time += time_unaccounted

            logger.logkv('Time/All_Training', all_time_training / time_total)
            logger.logkv('Time/All_Collection', all_time_collection / time_total)
            logger.logkv('Time/All_RPUse', all_rp_splice_time / time_total)
            logger.logkv('Time/All_RPTrain', all_time_rp_train / time_total)
            logger.logkv('Time/All_RunwTeacher', all_run_policy_time / time_total)
            logger.logkv('Time/All_Distillation', all_distill_time / time_total)
            logger.logkv('Time/All_VidRollout', all_rollout_time / time_total)
            logger.logkv('Time/All_Saving', all_saving_time / time_total)
            logger.logkv('Time/All_Unaccounted', all_unaccounted_time / time_total)

            for k in teacher_train_dict.keys():
                if should_train_rl:
                    logger.logkv(f'Feedback/Trained_{k}', int(teacher_train_dict[k]))
                else:
                    logger.logkv(f'Feedback/Trained_{k}', -1)

                if should_distill:
                    if self.args.distillation_strategy in ['all_teachers', 'all_but_none', 'powerset']:
                        logger.logkv(f'Feedback/Distilled_{k}', int(teacher_distill_dict[k]))
                else:
                    logger.logkv(f'Feedback/Distilled_{k}', -1)

                if should_policy_rollout:
                    logger.logkv(f'Feedback/Rollout_{k}', int(k in self.introduced_teachers))
                else:
                    logger.logkv(f'Feedback/Rollout_{k}', -1)

            logger.dumpkvs()

            """ ------------------ Video Saving ---------------------"""

            should_save_video = (itr % self.save_videos_every == 0) or (
                    itr == self.args.n_itr - 1)
            # If we're just collecting, don't log
            if self.args.no_train_rl and self.args.self_distill:
                should_save_video = False
            if self.args.yes_rollouts:
                should_save_video = True
            if self.args.no_rollouts:
                should_save_video = False
            if should_save_video:
                time_rollout_start = time.time()
                for teacher in self.introduced_teachers:
                    self.save_videos(self.policy_dict[teacher],
                                     save_name=f'{teacher}_video_stoch',
                                     num_rollouts=10,
                                     teacher_dict={k: k == teacher for k in teacher_train_dict.keys()},
                                     save_video=should_save_video,
                                     log_prefix=f"VidRollout/{teacher}_Stoch",
                                     teacher_name=teacher,
                                     stochastic=True,
                                     show_instrs=True if teacher == 'none' else not self.args.rollout_without_instrs)
                rollout_time = time.time() - time_rollout_start
            else:
                rollout_time = 0
            rollout_time = 0

            params = self.get_itr_snapshot(itr)
            step = self.curriculum_step

            early_stopping = self.itrs_since_best > self.args.early_stop
            best_success, best_accuracy, best_loss = self.best_train_perf
            # early_stopping = early_stopping and best_success > .7
            logger.logkv('Train/BestSuccess', best_success)
            logger.logkv('Train/BestAccuracy', best_accuracy)
            logger.logkv('Train/BestLoss', best_loss)
            logger.logkv('Train/ItrsSinceBest', self.itrs_since_best)


            if self.log_and_save:
                if early_stopping or (itr % self.save_every == 0) or (itr == self.args.n_itr - 1):
                    saving_time_start = time.time()
                    logger.log("Saving snapshot...")
                    logger.save_itr_params(itr, step, params)
                    logger.log("Saved")
                    saving_time = time.time() - saving_time_start

            if early_stopping:
                break

            if self.args.end_on_full_buffer:
                advance_curriculum = buffer.counts_train[self.curriculum_step] == buffer.train_buffer_capacity

            advance_curriculum = False
            if advance_curriculum:
                self.advancement_count += 1
            else:
                self.advancement_count = 0

            if self.advancement_count >= self.advancement_count_threshold:
                self.advancement_count = 0
                # if self.il_trainer is not None:
                #    self.run_with_bad_teachers(buffer, teacher_train_dict)
                # buffer.trim_level(self.curriculum_step, max_trajs=20000)
                # last_accuracy = 0
                # last_success = 0
                self.success_dict = {k: 0 for k in self.success_dict.keys()}
                self.curriculum_step += 1
                if self.curriculum_step >= len(self.env.train_levels):
                    break  # We've finished the curriculum!
                try:
                    self.sampler.advance_curriculum()
                    self.algo.advance_curriculum()
                except NotImplementedError:
                    # If we get a NotImplementedError b/c we ran out of levels, stop training
                    break
                self.itrs_on_level = 0
                self.next_train_itr = itr + 1
                self.num_train_skip_itrs = 5

        logger.log("Training finished")

    def evaluate_heldout(self, policy, teachers):
        num_rollouts = 1  # 50
        for i, env in enumerate(self.env.held_out_levels):
            level_name = env.__class__.__name__[6:]
            save_dir = pathlib.Path(self.exp_name)
            if not save_dir.exists():
                save_dir.mkdir()
            finetune_itrs = 0
            with open(save_dir.joinpath('results.csv'), 'w') as f:
                f.write('policy_env,policy, env,success_rate, stoch_accuracy, det_accuracy, followed_cc3 \n')
            try:
                teacher_null_dict = env.teacher.null_feedback()
            except Exception as e:
                teacher_null_dict = {}
            num_train_levels = len(self.env.train_levels)
            index = i + num_train_levels
            success_rate, stoch_accuracy, det_accuracy = test_success(env, index, save_dir, finetune_itrs, num_rollouts,
                                                                      teachers, teacher_null_dict, policy=policy,
                                                                      policy_name='latest', env_name=level_name)
            logger.logkv(f'Heldout/{level_name}Success', success_rate)
            logger.logkv(f'Heldout/{level_name}StochAcc', success_rate)
            logger.logkv(f'Heldout/{level_name}DetAcc', success_rate)

    def _log(self, episode_logs, summary_logs, data, tag=""):
        if episode_logs is not None:
            avg_return = np.mean(episode_logs['return_per_episode'])
            avg_path_length = np.mean(episode_logs['num_frames_per_episode'])
            avg_success = np.mean(episode_logs['success_per_episode'])
            avg_dist_to_goal = np.mean(episode_logs['dist_to_goal_per_episode'])
            avg_reward = np.mean(episode_logs["return_per_episode"])
            logger.logkv(f"{tag}/Success", avg_success)
            logger.logkv(f"{tag}/DistToGoal", avg_dist_to_goal)
            logger.logkv(f"{tag}/Reward", avg_reward)
            logger.logkv(f"{tag}/Return", avg_return)
            logger.logkv(f"{tag}/PathLength", avg_path_length)

            if self.args.discrete:
                logger.logkv(f"{tag}/Accuracy", torch.eq(data.action, data.teacher_action).float().mean().item())
                logger.logkv(f"{tag}/Argmax_Accuracy", torch.eq(data.action_probs.argmax(dim=1),
                                                                data.teacher_action).float().mean().item())

            self.num_feedback_advice += episode_logs['num_feedback_advice']
            self.num_feedback_reward += episode_logs['num_feedback_reward']
            for k in self.no_teacher_dict.keys():
                k_gave = f'gave_{k}'
                if k_gave in episode_logs:
                    self.gave_feedback[k] += episode_logs[k_gave]
                    logger.logkv(f"Feedback/Total_{k_gave}", self.gave_feedback[k])
                    k_followed = f'followed_{k}'
                    if k_followed in episode_logs:
                        self.followed_feedback[k] += episode_logs[k_followed]
                        logger.logkv(f"Feedback/Total_{k_followed}", self.followed_feedback[k])
                        logger.logkv(f"Feedback/Ratio_{k_followed}", episode_logs[k_followed] / episode_logs[k_gave])

            logger.logkv(f"{tag}/NumFeedbackAdvice", self.num_feedback_advice)
            logger.logkv(f"{tag}/NumFeedbackReward", self.num_feedback_reward)
            logger.logkv(f"{tag}/NumFeedbackTotal", self.num_feedback_advice + self.num_feedback_reward)
            logger.logkv(f"{tag}/num_feedback_reward", episode_logs['num_feedback_reward'])
            logger.logkv(f"{tag}/num_feedback_advice", episode_logs['num_feedback_advice'])
            for key in episode_logs:
                if 'followed_' in key or 'gave_' in key:
                    logger.logkv(f"Feedback/{key}", episode_logs[key])
        if summary_logs is not None:
            for k, v in summary_logs.items():
                if not k == 'Accuracy':
                    logger.logkv(f"{tag}/{k}", v)

    def use_reward_predictor(self, samples_data):
        pass
        # with torch.no_grad():
        #     r_discrete, logprobs = self.reward_predictor.get_actions(samples_data['env_infos']['next_obs_rewardfree'])
        #     if self.sparse_rewards:
        #         self.log_rew_pred(r_discrete[:, :, 0], samples_data['rewards'], samples_data['env_infos'])
        #     # Splice into the inference process
        #     if self.use_rp_inner:
        #         samples_data['observations'][:, :, -2] = r_discrete[:, :, 0]
        #     # Splice into the meta-learning process
        #     if self.use_rp_outer:
        #         samples_data['rewards'] = r_discrete[:, :, 0]
        #     if 'teacher_action' in samples_data['env_infos']:
        #         samples_data['env_infos']['teacher_action'] = samples_data['env_infos']['teacher_action'].astype(np.int32)
        #     return samples_data

    def save_data(self, data, itr):
        file_name = osp.join(self.exp_name, 'batch_%d.pkl' % itr)
        joblib.dump(data, file_name, compress=3)

    def train_rp(self, samples, is_training=False):
        log = self.rp_trainer.distill(samples, source=self.args.source, is_training=is_training)
        return log

    def run_with_bad_teachers(self, buffer, teachers_dict):
        # We pass in teachers one at a time and see what success rate we'd get if we shuffle that teacher
        teacher_names = [teacher for teacher in teachers_dict.keys() if teachers_dict[teacher]]
        for teacher in teacher_names:
            teacher_subset_dict = {}
            for k in teachers_dict.keys():
                teacher_subset_dict[k] = False
            teacher_subset_dict[teacher] = True

            # Shuffle teachers
            sampled_batch = buffer.sample(total_num_samples=self.args.batch_size, split='val')
            teacher_feedback = [obs_dict[teacher] for obs_dict in sampled_batch.obs]
            random.shuffle(teacher_feedback)
            for obs_dict, shuffled_obs in zip(sampled_batch.obs, teacher_feedback):
                obs_dict[teacher] = shuffled_obs
            log = self.il_trainer.distill(sampled_batch, source=self.args.source, is_training=False,
                                          teachers_dict=teacher_subset_dict, distill_target='all')
            log_dict = list(log.values())[0]
            logger.logkv(f"CheckTeachers/Shuffled_{teacher}_Accuracy", log_dict['Accuracy'])

            # CorrectTeacher, no inst
            sampled_batch = buffer.sample(total_num_samples=self.args.batch_size, split='val')
            for obs_dict in sampled_batch.obs:
                obs_dict['instr'] = [0] * len(obs_dict['instr'])
            log = self.il_trainer.distill(sampled_batch, source=self.args.source, is_training=False,
                                          teachers_dict=teacher_subset_dict, distill_target='all')
            log_dict = list(log.values())[0]
            logger.logkv(f"CheckTeachers/NoInstr_{teacher}_Accuracy", log_dict['Accuracy'])

    def relabel_goal(self, batch):
        # only works for ant and pm
        # Split into batches
        trajs = []
        end_idxs = torch.where(batch.full_done == 1)[0].detach().cpu().numpy() + 1
        start_idxs = np.concatenate([[0], end_idxs[:-1]])
        for start, end in zip(start_idxs, end_idxs):
            traj = batch[start: end]
            final_index = len(traj)
            if self.args.env == 'ant':
                assert self.args.show_pos == 'ours'
                goal_index = 29
                repeat_len = goal_index + 2
                goal_start_indices = np.arange(self.env.repeat_input) * repeat_len + goal_index
            elif self.args.env == 'point_mass':
                goal_index = 245
                repeat_len = 2
                goal_start_indices = np.arange(self.env.repeat_input) * repeat_len + goal_index
            else:
                raise NotImplementedError(type(self.env))

            # Relabel goal
            for i, obs_dict in enumerate(traj.obs):
                relabel_index = np.random.randint(i, final_index)
                last_obs = traj.obs[relabel_index]['obs']
                final_agent_pos = last_obs[:2]  # Note, it is scaled by a scaling factor
                o = obs_dict['obs']
                if self.args.env == 'point_mass':
                    goal = final_agent_pos
                elif self.env.args.show_goal == 'ours':
                    goal = final_agent_pos
                elif self.env.args.show_goal == 'offset':
                    curr_agent_pos = o[:2]
                    goal = final_agent_pos - curr_agent_pos
                else:
                    raise NotImplementedError
                for index in goal_start_indices:
                    o[index: index + 2] = goal
                obs_dict['obs'] = o
            trajs.append(traj)
        relabeled_batch = merge_dictlists(trajs)
        assert torch.all(torch.eq(relabeled_batch.action, batch.action))
        return relabeled_batch

    def relabel_reward(self, batch, relabel_dict):
        active_teachers = [k for k, v in relabel_dict.items() if v]
        teacher_name = 'none' if len(active_teachers) == 0 else active_teachers[0]
        obss, action_true, action_teacher = batch
        instr_dropout_prob = 0  # TODO: consider this!
        obss_processed = self.il_trainer_rew.preprocess_obs(obss, relabel_dict, show_instrs=np.random.uniform() > instr_dropout_prob)
        acmodel = self.il_trainer_rew.policy_dict[teacher_name]
        acmodel.eval()
        rew, _ = acmodel(obss_processed)
        rew = rew / 10  # we predict [0, 1], but give [0, .1]
        batch.reward = rew

    def distill(self, samples, is_training=False, teachers_dict=None, source=None, relabel=False, relabel_dict={},
                distill_to_none=True, train_reward=False):
        if source is None:
            source = self.args.source
        if train_reward:
            return self.il_trainer_rew.distill(samples, source=source, is_training=is_training,
                                           teachers_dict=teachers_dict, distill_target=self.args.distillation_strategy,
                                           relabel=relabel, relabel_dict=relabel_dict, distill_to_none=distill_to_none)
        log = self.il_trainer.distill(samples, source=source, is_training=is_training,
                                      teachers_dict=teachers_dict, distill_target=self.args.distillation_strategy,
                                      relabel=relabel, relabel_dict=relabel_dict, distill_to_none=distill_to_none)
        return log

    def run_supervised(self, policy, teacher_dict, tag, show_instrs):
        policy.eval()
        key_set = '_'.join([k for k, v in teacher_dict.items() if v])
        paths = self.sampler.obtain_samples(log=False, advance_curriculum=False, policy=policy,
                                            teacher_dict=teacher_dict, max_action=False, show_instrs=show_instrs,
                                            temperature=self.args.rollout_temperature)
        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix=tag + key_set,
                                                             log_teacher=self.train_with_teacher)
        use_teacher = not key_set == ''
        advance_curriculum, avg_success, avg_accuracy, avg_reward = \
            self.check_advance_curriculum_rollout(samples_data, use_teacher)
        logger.logkv(f"{tag}{key_set}Advance", int(advance_curriculum))
        logger.logkv(f"{tag}{key_set}AvgSuccess", avg_success)
        logger.logkv(f"{tag}{key_set}AvgAccuracy", avg_accuracy)
        logger.logkv(f"{tag}{key_set}AvgReward", avg_reward)
        logger.logkv(f"{tag}{key_set}FinalReward", np.mean(samples_data['rewards'][:, -1]))

        return advance_curriculum, avg_success, avg_accuracy

    def save_videos(self, policy, save_name='sample_video', num_rollouts=2, teacher_dict={}, save_video=False,
                    log_prefix=None, stochastic=True, teacher_name="", rollout_oracle=False, show_instrs=True):
        policy.eval()
        try:
            self.env.set_level_distribution(self.curriculum_step)
        except:
            print("no curriculum")
        save_wandb = (save_video and not self.is_debug)
        speedup = 1 if self.args.env == 'babyai' else 10
        paths, accuracy, stoch_accuracy, det_accuracy, cc3_followed = rollout(self.env, policy,
                                                                              reset_every=self.args.rollouts_per_meta_task,
                                                                              stochastic=stochastic,
                                                                              record_teacher=True,
                                                                              teacher_dict=teacher_dict,
                                                                              video_directory=self.exp_name,
                                                                              video_name=save_name + str(
                                                                                  self.curriculum_step),
                                                                              num_rollouts=num_rollouts,
                                                                              save_wandb=False,  # save_wandb,
                                                                              save_locally=True,
                                                                              num_save=20,
                                                                              obs_preprocessor=self.obs_preprocessor,
                                                                              teacher_name=teacher_name,
                                                                              rollout_oracle=rollout_oracle,
                                                                              instrs=show_instrs,
                                                                              temperature=self.args.rollout_temperature,
                                                                              speedup=speedup,
                                                                              hierarchical=self.args.hierarchical)
        if log_prefix is not None:
            logger.logkv(log_prefix + "Acc", accuracy)
            logger.logkv(log_prefix + "Stoch_Acc", stoch_accuracy)
            logger.logkv(log_prefix + "Det_Acc", det_accuracy)
            logger.logkv(log_prefix + "Reward", np.mean([sum(path['rewards']) for path in paths]))
            logger.logkv(log_prefix + "PathLength",
                         np.mean([path['env_infos'][-1]['episode_length'] for path in paths]))
            success = np.mean([path['env_infos'][-1]['success'] for path in paths])
            logger.logkv(log_prefix + "Success", success)
        return None

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        il_optimizer = self.il_trainer.optimizer_dict
        d = dict(itr=itr,
                 policy=self.policy_dict,
                 reward=self.rew_dict,
                 env=self.env,
                 args=self.args,
                 optimizer=self.algo.optimizer_dict,
                 reconstructor=self.algo.reconstructor_dict,
                 reconstructor_optimizer=self.algo.reconstructor_optimizer_dict,
                 il_reconstructor_optimizer=self.il_trainer.reconstructor_optimizer_dict,
                 curriculum_step=self.curriculum_step,
                 il_optimizer=il_optimizer,
                 log_dict={
                     'num_feedback_advice': self.num_feedback_advice,
                     'num_feedback_reward': self.num_feedback_reward,
                     'total_distillation_frames': self.total_distillation_frames,
                     'itrs_on_level': self.itrs_on_level,
                     'next_train_itr': self.next_train_itr,
                     'num_train_skip_itrs': self.num_train_skip_itrs,
                     'gave_feedback': self.gave_feedback,
                     'followed_feedback': self.followed_feedback,
                     'introduced_teachers': self.introduced_teachers,
                 })
        if self.reward_predictor is not None:
            d['reward_predictor'] = self.reward_predictor
        return d

    def _log_rew_pred(self, r_discrete, rewards, log_prefix):
        correct = rewards == r_discrete
        positives = rewards > 0
        pred_positives = r_discrete > 0
        negatives = 1 - positives
        incorrect = 1 - correct

        if positives.sum() > 0:
            false_negatives = (incorrect * positives).sum() / positives.sum()
            logger.logkv(log_prefix + 'FalseNegative', false_negatives)
            recall = (correct * positives).sum() / positives.sum()
            logger.logkv(log_prefix + 'Recall', recall)
        else:
            logger.logkv(log_prefix + 'FalseNegative', -1)  # Indicates none available

        if negatives.sum() > 0:
            false_positives = (incorrect * negatives).sum() / negatives.sum()
            logger.logkv(log_prefix + 'FalsePositive', false_positives)
        else:
            logger.logkv(log_prefix + 'FalsePositive', -1)

        if pred_positives.sum():
            precision = (correct * positives).sum() / pred_positives.sum()
            logger.logkv(log_prefix + 'Precision', precision)
        else:
            logger.logkv(log_prefix + 'Precision', -1)

        if positives.sum() > 0 and pred_positives.sum() > 0 and correct.sum() > 0:
            logger.logkv(log_prefix + 'F1', 2 * precision * recall / (precision + recall))
        else:
            logger.logkv(log_prefix + 'F1', -1)

    def log_rew_pred(self, r_discrete, rewards, env_infos):
        pass
        # TODO: Currently commented out since we compute this as if it was a binary classification, and we have a 3-way
        #   classification.  Still, we should rewrite this function to work with the new RP.
        # log_prefix = "RewPred/"
        #
        # # Flatten, trim out any which are just there for padding
        # # Elements where step=0 are just padding on the end.
        # curr_elements = 1 - (env_infos['step'].flatten() == 0)
        # r_discrete = np.stack([data for data, curr_bool in zip(r_discrete.flatten(), curr_elements) if curr_bool])
        # rewards = np.stack([data for data, curr_bool in zip(rewards.flatten(), curr_elements) if curr_bool])
        # step = np.stack([data for data, curr_bool in zip(env_infos['step'].flatten(), curr_elements) if curr_bool])
        #
        #
        # self._log_rew_pred(r_discrete, rewards, log_prefix)

        # # Log split by index in meta-task
        # unique_steps = np.unique(env_infos['step'])
        # for i in unique_steps:
        #     if i == 0:
        #         continue  # Remove 0, which is just filler
        #     curr_elements = step == i
        #     r_discrete_i = np.stack([data for data, curr_bool in zip(r_discrete, curr_elements) if curr_bool])
        #     rewards_i = np.stack([data for data, curr_bool in zip(rewards, curr_elements) if curr_bool])
        #     self._log_rew_pred(r_discrete_i, rewards_i, log_prefix + str(i) + "-")
