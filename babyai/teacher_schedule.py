import copy


# A distillation scheme is represented by a function which takes in a level number and returns a dictionary specifying
# What teachers the agent should train on and distill to
# For the moment, we hard-code the levels, although later it might be nice to determine which teachers to use
# Based on the success/reward from the past itr

#### NO TEACHER ####
def no_teacher(level):
    return {}, {}


#### SINGLE TEACHER ####
def all_teachers(level, teacher_list):
    no_teacher_dict = {}
    teacher_train_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
        teacher_train_dict[teacher] = True
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    distillation_dict = copy.deepcopy(teacher_train_dict)
    return teacher_train_dict, distillation_dict


#### FIRST TEACHER ####
# Train on the first teacher, distill to all the others
def first_teacher(level, teacher_list):
    no_teacher_dict = {}
    distillation_dict = {}
    for teacher in teacher_list:
        no_teacher_dict[teacher] = False
        distillation_dict[teacher] = True
    if level == -1:  # Generate no_teacher_dict
        return no_teacher_dict, None
    teacher_train_dict = copy.deepcopy(no_teacher_dict)
    teacher_train_dict[teacher_list[0]] = True
    return teacher_train_dict, distillation_dict

#### SINGLE TEACHER, NO POWERSET ####
def single_teacher_no_powerset(level, teacher_name):
    if level == -1:  # Generate no_teacher_dict
        return {teacher_name: False}, None
    teacher_train_dict = {teacher_name: True}
    distillation_dict = {teacher_name: False}
    return teacher_train_dict, distillation_dict


### PREACTION TO ONE OTHER ####
# Add in the second teacher ...
def easy_add_harder(level, easy_teacher, harder_teacher, cutoff_level=21):
    if level == -1:  # Generate no_teacher_dict
        return {easy_teacher: False, harder_teacher: False}, None
    if level < cutoff_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
    else:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
    distillation_dict = copy.deepcopy(teacher_train_dict)
    return teacher_train_dict, distillation_dict


### PREACTION TO ONE OTHER, SWAP OUT ####
# Add in the second teacher ...
def easy_replace_harder(level, easy_teacher, harder_teacher, add_hard_level=15, remove_easy_level=21):
    if level == -1:  # Generate no_teacher_dict
        return {easy_teacher: False, harder_teacher: False}, None
    if level < add_hard_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: False}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    elif level < remove_easy_level:
        teacher_train_dict = {easy_teacher: True, harder_teacher: True}
        distillation_dict = copy.deepcopy(teacher_train_dict)
    else:
        teacher_train_dict = {easy_teacher: False, harder_teacher: True}
        distillation_dict = {easy_teacher: True, harder_teacher: True}
    return teacher_train_dict, distillation_dict


def make_teacher_schedule(feedback_types, teacher_schedule):
    feedback_types = [teacher for teacher in feedback_types if not teacher == 'None']
    if teacher_schedule == 'none':
        assert len(feedback_types) == 0
        return no_teacher
    elif teacher_schedule == 'all_teachers':
        return lambda level: all_teachers(level, feedback_types)
    elif teacher_schedule == 'first_teacher':
        return lambda level: first_teacher(level, feedback_types)
    elif teacher_schedule == 'single_teacher_no_powerset':
        assert len(feedback_types) == 1
        return lambda level: single_teacher_no_powerset(level, feedback_types[0])
    elif teacher_schedule == 'easy_add_harder':
        assert len(feedback_types) == 2
        return lambda level: easy_add_harder(level, feedback_types[0], feedback_types[1])
    elif teacher_schedule == 'easy_swap_harder':
        assert len(feedback_types) == 2
        return lambda level: easy_add_harder(level, feedback_types[0], feedback_types[1])
    else:
        raise ValueError(f'Unknown distillation scheme {teacher_schedule}, {feedback_types}')