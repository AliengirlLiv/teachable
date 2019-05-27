import gym
import operator
from gym_minigrid.envs import Key, Ball, Box
from .verifier import *
from .levelgen import *


class Level_OpenRedDoor(RoomGridLevel):
    """
    Go to the red door
    (always unlocked, in the current room)
    Note: this level is intentionally meant for debugging and is
    intentionally kept very simple.
    """

    def __init__(self, seed=None):
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=5,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_door(0, 0, 0, 'red', locked=False)
        self.place_agent(0, 0)
        self.instrs = OpenInstr(ObjDesc('door', 'red'))


class Level_OpenDoor(RoomGridLevel):
    """
    Go to the door
    The door to open is given by its color or by its location.
    (always unlocked, in the current room)
    """

    def __init__(
        self,
        debug=False,
        select_by=None,
        seed=None
    ):
        self.select_by = select_by
        self.debug = debug
        super().__init__(seed=seed)

    def gen_mission(self):
        door_colors = self._rand_subset(COLOR_NAMES, 4)
        objs = []

        for i, color in enumerate(door_colors):
            obj, _ = self.add_door(1, 1, door_idx=i, color=color, locked=False)
            objs.append(obj)

        select_by = self.select_by
        if select_by is None:
            select_by = self._rand_elem(["color", "loc"])
        if select_by == "color":
            object = ObjDesc(objs[0].type, color=objs[0].color)
        elif select_by == "loc":
            object = ObjDesc(objs[0].type, loc=self._rand_elem(LOC_NAMES))

        self.place_agent(1, 1)
        self.instrs = OpenInstr(object, strict=self.debug)


class Level_OpenDoorDebug(Level_OpenDoor):
    """
    Same as OpenDoor but the level stops when any door is opened
    """

    def __init__(
        self,
        select_by=None,
        seed=None
    ):
        super().__init__(select_by=select_by, debug=True, seed=seed)


class Level_OpenDoorColor(Level_OpenDoor):
    """
    Go to the door
    The door is selected by color.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="color",
            seed=seed
        )


class Level_OpenDoorLoc(Level_OpenDoor):
    """
    Go to the door
    The door is selected by location.
    (always unlocked, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            select_by="loc",
            seed=seed
        )


class Level_GoToDoor(RoomGridLevel):
    """
    Go to a door
    (of a given color, in the current room)
    No distractors, no language variation
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        objs = []
        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)
        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        self.instrs = GoToInstr(ObjDesc('door', obj.color))


class Level_GoToObjDoor(RoomGridLevel):
    """
    Go to an object or door
    (of a given type and color, in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            seed=seed,
            max_steps=64
        )

    def get_objs(self):
        'find obj in environment and create description'
        self.place_agent(1, 1)
        objs = self.add_distractors(1, 1, num_distractors=8, all_unique=False)

        for _ in range(4):
            door, _ = self.add_door(1, 1)
            objs.append(door)

        self.check_objs_reachable()
        return objs

    def reset(self, **kwargs):
        'override reset to preserve max_steps=64'
        obs = super().reset(**kwargs)
        self.max_steps = 64
        return obs

    def get_ObjDesc(self, objs):
        'select obj to goto'
        obj = self._rand_elem(objs)
        return ObjDesc(obj.type, obj.color)

    def gen_mission(self):
        'create instruction from description'
        objs = self.get_objs()
        objDesc = self.get_ObjDesc(objs)
        self.instrs = GoToInstr(objDesc)


class Level_ControllerAllTasks(RoomGridLevel):
    """
    <explore> or <go (next) to obj / door>
    from inside a room or from a door facing the room
    such that all possible interpretations of the instruction are equiprobable
    """

    def __init__(self, seed=None):
        self.loc = False
        self.carryInv = True
        super().__init__(
            room_size=8,
            seed=seed,
            max_steps=64
        )
        self.doNotOpenBox = True
        self.doNotOpenDoor = True

    def reset(self, **kwargs):
        'override reset to preserve max_steps=64'
        obs = super().reset(**kwargs)
        self.max_steps = 64
        return obs

    def place_agent(self, rand_dir=True):
        'place agent in centremost room or facing a door inwards to the room'
        if self._rand_bool():
            return super().place_agent(1, 1), True
        # place outside the room
        doors = zip(self.room.doors, range(4))
        doors = list(filter(lambda d: d[0] is not None, doors))

        door, dir = self._rand_elem(doors)
        door.is_open = True
        pos = door.cur_pos
        dir = (dir + 2) % 4
        vec = DIR_TO_VEC[dir]

        self.start_pos = pos - vec
        self.start_dir = dir
        return self.start_pos, False

    def add_obstructors(self):
        'add obstructors to environment'
        for i in range(3):
            for j in range(3):
                num_dist = self._rand_int(0, 9)
                self.add_distractors(i, j, num_distractors=num_dist, all_unique=False)

    def get_objs(self):
        'find obj in environment and create description'
        self.connect_all()
        self.room = self.get_room(1, 1)
        doors = list(filter(lambda d: d is not None, self.room.doors))
        for door in doors:
            door.is_open = self._rand_bool()
        pos, self.center = self.place_agent()
        self.pos = tuple(pos)
        self.doors = list(filter(lambda d: not pos_next_to(d.cur_pos, pos), doors))
        self.add_obstructors()

    def carrying_object(self):
        'randomly choose if agent should carry, if so, randomly generate object'
        if self._rand_bool():
            object = self._rand_elem([Key, Ball, Box])
            color = self._rand_color()
            obj = object(color)
            return obj

    def get_loc(self, obj):
        'get location of an object'
        i, j = obj.cur_pos
        x, y = self.pos
        v = (i-x, j-y)
        # (d1, d2) is an oriented orthonormal basis
        d1 = DIR_TO_VEC[self.start_dir]
        d2 = (-d1[1], d1[0])
        # Check if object's position matches with location
        pos_matches = {
            "left": -dot_product(v, d2),
            "right": dot_product(v, d2),
            "front": dot_product(v, d1),
            "behind": -dot_product(v, d1)
        }
        return max(pos_matches.items(), key=operator.itemgetter(1))[0]

    def create_desc(self, objs):
        'randomly select object to go to'
        obj = self._rand_elem(objs)
        colors = [o.color for o in objs if o.color == obj.color]
        loc = None
        if self.loc and (self._rand_bool() or len(colors) > 1):
            loc = self.get_loc(obj)
        objDesc = ObjDesc(obj.type, obj.color, loc)
        return objDesc

    def obj_in_room(self):
        'if no obj in room, add'
        if len(self.room.objs) == 0:
            self.add_distractors(1, 1, num_distractors=1, all_unique=False)

    def door_in_room(self):
        'if no door in room, add'
        while len(self.doors) == 0:
            self.add_door(1, 1)
            doors = list(filter(lambda d: d is not None, self.room.doors))
            self.doors = list(filter(lambda d: not pos_next_to(d.cur_pos, self.pos), doors))

    def gen_instr_type(self, carry):
        'generate a random instruction type'
        self.get_objs()
        instrType = self._rand_int(0, 4)
        if instrType == 0:
            # create door if none exists then go to
            self.door_in_room()
            objDesc = self.create_desc(self.doors)
            self.instrs = GoToInstr(objDesc, **carry)
        elif instrType == 1:
            # create object if none exists then go to
            self.obj_in_room()
            objDesc = self.create_desc(self.room.objs)
            self.instrs = GoToInstr(objDesc, **carry)
        elif instrType == 2:
            # create object if none exists then go next to
            self.obj_in_room()
            objDesc = self.create_desc(self.room.objs)
            self.instrs = GoNextToInstr(objDesc, **carry, objs=self.room.objs)
        else:
            # otherwise explore
            self.instrs = ExploreInstr()

    def gen_mission(self):
        'create instruction from description'
        self.carrying = self.carrying_object()
        carry = dict(carrying=self.carrying, carryInv=self.carryInv, middle=True)
        self.gen_instr_type(carry)


class Level_ControllerAllGoToTasks(Level_ControllerAllTasks):
    """
    go (next) to obj / door
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_instr_type(self, carry):
        'generate a random instruction type'
        self.get_objs()
        instrType = self._rand_int(0, 3)
        if instrType == 0:
            # create door if none exists then go to
            self.door_in_room()
            objDesc = self.create_desc(self.doors)
            self.instrs = GoToInstr(objDesc, **carry)
        else:
            # create object if none exists then go to
            self.obj_in_room()
            objDesc = self.create_desc(self.room.objs)
            if instrType == 1:
                self.instrs = GoToInstr(objDesc, **carry)
            else:
                self.instrs = GoNextToInstr(objDesc, **carry, objs=self.room.objs)


class Level_ControllerExplore(Level_ControllerAllTasks):
    """
    explore the middle room
    this level and its descendents are for testing
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        'explore'
        self.get_objs()
        self.carrying = self.carrying_object()
        self.instrs = ExploreInstr(carrying=self.carrying, carryInv=self.carryInv)


class Level_ControllerGoTo(Level_ControllerExplore):
    """
    go to an object in the middle room
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        'if object present, goto, otherwise explore'
        self.get_objs()
        self.carrying = self.carrying_object()
        carry = dict(carrying=self.carrying, carryInv=self.carryInv, middle=True)
        self.obj_in_room()
        objDesc = self.create_desc(self.room.objs)
        self.instrs = GoToInstr(objDesc, **carry)


class Level_ControllerGoToDoor(Level_ControllerExplore):
    """
    go to a door in the middle room
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        'if door present, goto, otherwise explore'
        self.get_objs()
        self.carrying = self.carrying_object()
        carry = dict(carrying=self.carrying, carryInv=self.carryInv, middle=True)
        self.door_in_room()
        objDesc = self.create_desc(self.doors)
        self.instrs = GoToInstr(objDesc, **carry)


class Level_ControllerGoNextTo(Level_ControllerExplore):
    """
    go next to an object in the middle room
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        'if object present, go next to, otherwise explore'
        self.get_objs()
        self.carrying = self.carrying_object()
        carry = dict(carrying=self.carrying, carryInv=self.carryInv, objs=self.room.objs, middle=True)
        self.obj_in_room()
        objDesc = self.create_desc(self.room.objs)
        self.instrs = GoNextToInstr(objDesc, **carry)


class Level_ActionObjDoor(RoomGridLevel):
    """
    [pick up an object] or
    [go to an object or door] or
    [open a door]
    (in the current room)
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        objs = self.add_distractors(1, 1, num_distractors=5)
        for _ in range(4):
            door, _ = self.add_door(1, 1, locked=False)
            objs.append(door)

        self.place_agent(1, 1)

        obj = self._rand_elem(objs)
        desc = ObjDesc(obj.type, obj.color)

        if obj.type == 'door':
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = OpenInstr(desc)
        else:
            if self._rand_bool():
                self.instrs = GoToInstr(desc)
            else:
                self.instrs = PickupInstr(desc)


class Level_UnlockLocal(RoomGridLevel):
    """
    Fetch a key and unlock a door
    (in the current room)
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors
        super().__init__(seed=seed)

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)
        self.add_object(1, 1, 'key', door.color)
        if self.distractors:
            self.add_distractors(1, 1, num_distractors=3)
        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class Level_UnlockLocalDist(Level_UnlockLocal):
    """
    Fetch a key and unlock a door
    (in the current room, with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)


class Level_KeyInBox(RoomGridLevel):
    """
    Unlock a door. Key is in a box (in the current room).
    """

    def __init__(self, seed=None):
        super().__init__(
            seed=seed
        )

    def gen_mission(self):
        door, _ = self.add_door(1, 1, locked=True)

        # Put the key in the box, then place the box in the room
        key = Key(door.color)
        box = Box(self._rand_color(), key)
        self.place_in_room(1, 1, box)

        self.place_agent(1, 1)

        self.instrs = OpenInstr(ObjDesc(door.type))


class Level_UnlockPickup(RoomGridLevel):
    """
    Unlock a door, then pick up a box in another room
    """

    def __init__(self, distractors=False, seed=None):
        self.distractors = distractors

        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)
        if self.distractors:
            self.add_distractors(num_distractors=4)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_UnlockPickupDist(Level_UnlockPickup):
    """
    Unlock a door, then pick up an object in another room
    (with distractors)
    """

    def __init__(self, seed=None):
        super().__init__(distractors=True, seed=seed)


class Level_BlockedUnlockPickup(RoomGridLevel):
    """
    Unlock a door blocked by a ball, then pick up a box
    in another room
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0]-1, pos[1], Ball(color))
        # Add a key to unlock the door
        self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_UnlockToUnlock(RoomGridLevel):
    """
    Unlock a door A that requires to unlock a door B before
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=30*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        # Add a door of color A connecting left and middle room
        self.add_door(0, 0, door_idx=0, color=colors[0], locked=True)

        # Add a key of color A in the room on the right
        self.add_object(2, 0, kind="key", color=colors[0])

        # Add a door of color B connecting middle and right room
        self.add_door(1, 0, door_idx=0, color=colors[1], locked=True)

        # Add a key of color B in the middle room
        self.add_object(1, 0, kind="key", color=colors[1])

        obj, _ = self.add_object(0, 0, kind="ball")

        self.place_agent(1, 0)

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_PickupDist(RoomGridLevel):
    """
    Pick up an object
    The object to pick up is given by its type only, or
    by its color, or by its type and color.
    (in the current room, with distractors)
    """

    def __init__(self, debug=False, seed=None):
        self.debug = debug
        super().__init__(
            num_rows = 1,
            num_cols = 1,
            room_size=7,
            seed=seed
        )

    def gen_mission(self):
        # Add 5 random objects in the room
        objs = self.add_distractors(num_distractors=5)
        self.place_agent(0, 0)
        obj = self._rand_elem(objs)
        type = obj.type
        color = obj.color

        select_by = self._rand_elem(["type", "color", "both"])
        if select_by == "color":
            type = None
        elif select_by == "type":
            color = None

        self.instrs = PickupInstr(ObjDesc(type, color), strict=self.debug)


class Level_PickupDistDebug(Level_PickupDist):
    """
    Same as PickupDist but the level stops when any object is picked
    """

    def __init__(self, seed=None):
        super().__init__(
            debug=True,
            seed=seed
        )


class Level_PickupAbove(RoomGridLevel):
    """
    Pick up an object (in the room above)
    This task requires to use the compass to be solved effectively.
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to the top-middle room
        obj, pos = self.add_object(1, 0)
        # Make sure the two rooms are directly connected
        self.add_door(1, 1, 3, locked=False)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))


class Level_OpenTwoDoors(RoomGridLevel):
    """
    Open door X, then open door Y
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self,
        first_color=None,
        second_color=None,
        strict=False,
        seed=None
    ):
        self.first_color = first_color
        self.second_color = second_color
        self.strict = strict

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, 2)

        first_color = self.first_color
        if first_color is None:
            first_color = colors[0]
        second_color = self.second_color
        if second_color is None:
            second_color = colors[1]

        door1, _ = self.add_door(1, 1, 2, color=first_color, locked=False)
        door2, _ = self.add_door(1, 1, 0, color=second_color, locked=False)

        self.place_agent(1, 1)

        self.instrs = BeforeInstr(
            OpenInstr(ObjDesc(door1.type, door1.color), strict=self.strict),
            OpenInstr(ObjDesc(door2.type, door2.color))
        )


class Level_OpenTwoDoorsDebug(Level_OpenTwoDoors):
    """
    Same as OpenTwoDoors but the level stops when the second door is opened
    """

    def __init__(self,
        first_color=None,
        second_color=None,
        seed=None
    ):
        super().__init__(
            first_color,
            second_color,
            strict=True,
            seed=seed
        )


class Level_OpenRedBlueDoors(Level_OpenTwoDoors):
    """
    Open red door, then open blue door
    The two doors are facing opposite directions, so that the agent
    Can't see whether the door behind him is open.
    This task requires memory (recurrent policy) to be solved effectively.
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )


class Level_OpenRedBlueDoorsDebug(Level_OpenTwoDoorsDebug):
    """
    Same as OpenRedBlueDoors but the level stops when the blue door is opened
    """

    def __init__(self, seed=None):
        super().__init__(
            first_color="red",
            second_color="blue",
            seed=seed
        )


class Level_FindObjS5(RoomGridLevel):
    """
    Pick up an object (in a random room)
    Rooms have a size of 5
    This level requires potentially exhaustive exploration
    """

    def __init__(self, room_size=5, seed=None):
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        # Add a random object to a random room
        i = self._rand_int(0, self.num_rows)
        j = self._rand_int(0, self.num_cols)
        obj, _ = self.add_object(i, j)
        self.place_agent(1, 1)
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_FindObjS6(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 6
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            seed=seed
        )


class Level_FindObjS7(Level_FindObjS5):
    """
    Same as the FindObjS5 level, but rooms have a size of 7
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            seed=seed
        )


class KeyCorridor(RoomGridLevel):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
        )

    def gen_mission(self):
        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_KeyCorridorS3R1(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=1,
            seed=seed
        )

class Level_KeyCorridorS3R2(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=2,
            seed=seed
        )

class Level_KeyCorridorS3R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS4R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS5R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed
        )

class Level_KeyCorridorS6R3(KeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed
        )

class Level_1RoomS8(RoomGridLevel):
    """
    Pick up the ball
    Rooms have a size of 8
    """

    def __init__(self, room_size=8, seed=None):
        super().__init__(
            room_size=room_size,
            num_rows=1,
            num_cols=1,
            seed=seed
        )

    def gen_mission(self):
        obj, _ = self.add_object(0, 0, kind="ball")
        self.place_agent()
        self.instrs = PickupInstr(ObjDesc(obj.type))


class Level_1RoomS12(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 12
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=12,
            seed=seed
        )


class Level_1RoomS16(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 16
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=16,
            seed=seed
        )


class Level_1RoomS20(Level_1RoomS8):
    """
    Pick up the ball
    Rooms have a size of 20
    """

    def __init__(self, seed=None):
        super().__init__(
            room_size=20,
            seed=seed
        )


class PutNext(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        start_carrying=False,
        seed=None
    ):
        assert room_size >= 4
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room
        self.start_carrying = start_carrying

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        a = self._rand_elem(objs_l)
        b = self._rand_elem(objs_r)

        # Randomly flip the object to be moved
        if self._rand_bool():
            t = a
            a = b
            b = t

        self.obj_a = a

        self.instrs = PutNextInstr(
            ObjDesc(a.type, a.color),
            ObjDesc(b.type, b.color)
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # If the agent starts off carrying the object
        if self.start_carrying:
            self.grid.set(*self.obj_a.init_pos, None)
            self.carrying = self.obj_a

        return obs


class Level_PutNextS4N1(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            objs_per_room=1,
            seed=seed
        )


class Level_PutNextS5N1(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=1,
            seed=seed
        )


class Level_PutNextS5N2(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            seed=seed
        )


class Level_PutNextS6N3(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            objs_per_room=3,
            seed=seed
        )


class Level_PutNextS7N4(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            objs_per_room=4,
            seed=seed
        )


class Level_PutNextS5N2Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            start_carrying=True,
            seed=seed
        )


class Level_PutNextS6N3Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            objs_per_room=3,
            start_carrying=True,
            seed=seed
        )


class Level_PutNextS7N4Carrying(PutNext):
    def __init__(self, seed=None):
        super().__init__(
            room_size=7,
            objs_per_room=4,
            start_carrying=True,
            seed=seed
        )


class MoveTwoAcross(RoomGridLevel):
    """
    Task of the form: move the A next to the B and the C next to the D.
    This task is structured to have a very large number of possible
    instructions.
    """

    def __init__(
        self,
        room_size,
        objs_per_room,
        seed=None
    ):
        assert objs_per_room <= 9
        self.objs_per_room = objs_per_room

        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=16*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        self.place_agent(0, 0)

        # Add objects to both the left and right rooms
        # so that we know that we have two non-adjacent set of objects
        objs_l = self.add_distractors(0, 0, self.objs_per_room)
        objs_r = self.add_distractors(1, 0, self.objs_per_room)

        # Remove the wall between the two rooms
        self.remove_wall(0, 0, 0)

        # Select objects from both subsets
        objs_l = self._rand_subset(objs_l, 2)
        objs_r = self._rand_subset(objs_r, 2)
        a = objs_l[0]
        b = objs_r[0]
        c = objs_r[1]
        d = objs_l[1]

        self.instrs = BeforeInstr(
            PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color)),
            PutNextInstr(ObjDesc(c.type, c.color), ObjDesc(d.type, d.color))
        )


class Level_MoveTwoAcrossS5N2(MoveTwoAcross):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            objs_per_room=2,
            seed=seed
        )


class Level_MoveTwoAcrossS8N9(MoveTwoAcross):
    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            objs_per_room=9,
            seed=seed
        )


class OpenDoorsOrder(RoomGridLevel):
    """
    Open one or two doors in the order specified.
    """

    def __init__(
        self,
        num_doors,
        debug=False,
        seed=None
    ):
        assert num_doors >= 2
        self.num_doors = num_doors
        self.debug = debug

        room_size = 6
        super().__init__(
            room_size=room_size,
            max_steps=20*room_size**2,
            seed=seed
        )

    def gen_mission(self):
        colors = self._rand_subset(COLOR_NAMES, self.num_doors)
        doors = []
        for i in range(self.num_doors):
            door, _ = self.add_door(1, 1, color=colors[i], locked=False)
            doors.append(door)
        self.place_agent(1, 1)

        door1, door2 = self._rand_subset(doors, 2)
        desc1 = ObjDesc(door1.type, door1.color)
        desc2 = ObjDesc(door2.type, door2.color)

        mode = self._rand_int(0, 3)
        if mode == 0:
            self.instrs = OpenInstr(desc1, strict=self.debug)
        elif mode == 1:
            self.instrs = BeforeInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        elif mode == 2:
            self.instrs = AfterInstr(OpenInstr(desc1, strict=self.debug), OpenInstr(desc2, strict=self.debug))
        else:
            assert False

class Level_OpenDoorsOrderN2(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=2,
            seed=seed
        )


class Level_OpenDoorsOrderN4(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=4,
            seed=seed
        )


class Level_OpenDoorsOrderN2Debug(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=2,
            debug=True,
            seed=seed
        )


class Level_OpenDoorsOrderN4Debug(OpenDoorsOrder):
    def __init__(self, seed=None):
        super().__init__(
            num_doors=4,
            debug=True,
            seed=seed
        )

for name, level in list(globals().items()):
    if name.startswith('Level_'):
        level.is_bonus = True

# Register the levels in this file
register_levels(__name__, globals())
