from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Set

from dataclasses_json import DataClassJsonMixin, config


@dataclass
class DataClassWithId(DataClassJsonMixin):
    id: str


@dataclass
class Patient(DataClassWithId):
    gender: str
    age_group: str
    length_of_stay: int
    workload_produced: List[int]
    skill_level_required: List[int]


@dataclass
class Occupant(Patient):
    room_id: str


@dataclass
class NewPatient(Patient):
    mandatory: bool
    surgery_release_day: int
    surgery_duration: int
    surgeon_id: str
    incompatible_room_ids: List[str]
    surgery_due_day: Optional[int] = field(default=None)


@dataclass
class Surgeon(DataClassWithId):
    max_surgery_time: List[int]


@dataclass
class OperatingTheater(DataClassWithId):
    availability: List[int]


@dataclass
class Room(DataClassWithId):
    capacity: int


@dataclass(frozen=True)
class WorkingShift(DataClassJsonMixin):
    day: int
    shift: str
    max_load: int


@dataclass
class NurseJSON(DataClassWithId):
    skill_level: int
    working_shifts: List[WorkingShift]


@dataclass
class Nurse:

    id: str
    skill_level: int
    working_shifts: Dict[int, str]
    working_loads: Dict[int, int]

    def __init__(self, json: NurseJSON) -> None:
        self.id = json.id
        self.skill_level = json.skill_level
        self.working_shifts = {s.day: s.shift for s in json.working_shifts}
        self.working_loads = {s.day: s.max_load for s in json.working_shifts}


@dataclass(frozen=True)
class Weights(DataClassJsonMixin):
    room_mixed_age: int
    room_nurse_skill: int
    continuity_of_care: int
    nurse_eccessive_workload: int
    open_operating_theater: int
    surgeon_transfer: int
    patient_delay: int
    unscheduled_optional: int


def lst_to_dict(transform: Callable[..., DataClassWithId], lst: List) -> Dict:
    return {e.id: e for e in [transform(d) for d in lst]}


@dataclass
class Instance(DataClassJsonMixin):

    days: int
    skill_levels: int
    shift_types: List[str]
    age_groups: List[str]
    occupants: Dict[str, Occupant] = field(metadata=config(decoder=lambda lst: lst_to_dict(Occupant.from_dict, lst)))
    patients: Dict[str, NewPatient] = field(metadata=config(decoder=lambda lst: lst_to_dict(NewPatient.from_dict, lst)))
    surgeons: Dict[str, Surgeon] = field(metadata=config(decoder=lambda lst: lst_to_dict(Surgeon.from_dict, lst)))
    operating_theaters: Dict[str, OperatingTheater] = field(metadata=config(
        decoder=lambda lst: lst_to_dict(OperatingTheater.from_dict, lst)))
    rooms: Dict[str, Room] = field(metadata=config(decoder=lambda lst: lst_to_dict(Room.from_dict, lst)))
    nurses: Dict[str, Nurse] = field(metadata=config(
        decoder=lambda lst: {n.id: n for n in [Nurse(NurseJSON.from_dict(obj)) for obj in lst]}))
    weights: Weights
    instance_file: Optional[str] = None

    @staticmethod
    def from_file(instance_file: str) -> 'Instance':
        with open(instance_file) as f:
            instance = Instance.from_json(f.read())
            instance.instance_file = instance_file
            return instance
