import sys
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict

import pandas as pd
from colour import Color
from great_tables import GT, style, loc
from dataclasses_json import DataClassJsonMixin, config

from Instance import Instance, NewPatient


@dataclass
class PatientAssignment(DataClassJsonMixin):
    id: str
    admission_day: Optional[int] = field(default=None,
                                         metadata=config(encoder=lambda d: 'none' if d is None else d,
                                                         decoder=lambda d: None if d == 'none' else int(d)))
    room: Optional[str] = None
    operating_theater: Optional[str] = None  # Ignore


@dataclass
class NurseAssignment(DataClassJsonMixin):
    day: int
    shift: str
    rooms: List[str]


@dataclass
class NurseSchedule(DataClassJsonMixin):
    id: str
    assignments: List[NurseAssignment]


@dataclass
class SolutionJSON(DataClassJsonMixin):
    patients: list[PatientAssignment]
    nurses: list[NurseSchedule]


class Solution:
    """Solution containing assignments of patients to days, rooms, and OTs; and of nurses to rooms.
    
    You can ignore nurse assignments and OT assignments.
    """

    def __init__(self, instance: Instance, from_json: Optional[SolutionJSON] = None):
        self.instance = instance
        if from_json is None:
            self.patients = {p: PatientAssignment(p) for p in instance.patients}
            self.nurses = {i: {d: set() for d in n.working_shifts} for i, n in instance.nurses.items()}  # Ignore
        else:
            self.patients = {p.id: p for p in from_json.patients}
            self.nurses = {n.id: {a.day: set(a.rooms) for a in n.assignments} for n in from_json.nurses}  # Ignore

    @staticmethod
    def from_file(instance: Instance, solution_file) -> 'Solution':
        with open(solution_file) as f:
            return Solution(instance, SolutionJSON.from_json(f.read()))

    def to_output(self) -> SolutionJSON:
        return SolutionJSON(list(self.patients.values()),
                            [NurseSchedule(i, [NurseAssignment(day, self.instance.nurses[i].working_shifts[day],
                                                               list(rooms))
                                               for day, rooms in schedule.items()])
                             for i, schedule in self.nurses.items()])

    def to_file(self, solution_file) -> None:
        with open(solution_file, 'w') as f:
            f.write(self.to_output().to_json(indent=2))

    @staticmethod
    def __find_index(data: List[List[Optional[Any]]], slot: int, length: int) -> int:
        index = len(data)
        for i in range(len(data)):
            if data[i][slot] is None:
                index = i
                break
        if index == len(data):
            data.append([None] * length)
        return index

    @staticmethod
    def __extract_assignments(data_dict: Dict[str, List[Optional[str]]], mapping: Dict[str, List[List[Optional[str]]]],
                              format_str: str, format_first: Optional[str] = None,
                              format_last: Optional[str] = None) -> None:
        for a, data in mapping.items():
            for n, content in enumerate(data):
                if n == 0 and format_first is not None:
                    label = format_first.format(a=a, n=n)
                elif n == len(data) - 1 and format_last is not None:
                    label = format_last.format(a=a, n=n)
                else:
                    label = format_str.format(a=a, n=n)
                data_dict[label] = content + [a]

    def print_table(self, details: bool = False) -> None:
        
        room_patient = {i: [] for i in self.instance.rooms}
        surgeon_patient = {i: [[None for _ in range(self.instance.days)]] for i in self.instance.surgeons}
        for i in self.instance.surgeons:
            for d in range(self.instance.days):
                s_sum = sum(self.instance.patients[o].surgery_duration for o, p in self.patients.items()
                            if p.admission_day == d and self.instance.patients[o].surgeon_id == i)
                surgeon_patient[i][0][d] = f'{s_sum}/{self.instance.surgeons[i].max_surgery_time[d]}'

        unscheduled_patients = []

        admissions = [(o, 0, o.room_id, None) for o in self.instance.occupants.values()]
        for p in sorted([p for p in self.patients.values() if p.admission_day is not None],
                        key=lambda x: (x.admission_day, x.id)):
            admissions.append((self.instance.patients[p.id], p.admission_day, p.room, None))

        color_map = {}

        def contrast_color(col: Color) -> Color:
            """Return the appropriate text color (black or white) for the given background color."""
            if 0.299 * col.red + 0.587 * col.green + 0.114 * col.blue > 0.5:
                return Color('black')
            else:
                return Color('white')

        # Patient to room assignments
        for patient, admission, room, _ in admissions:

            index = self.__find_index(room_patient[room], admission, self.instance.days)
            for i in range(admission, admission + patient.length_of_stay):
                if i >= self.instance.days:
                    break
                if details:
                    room_patient[room][index][i] = f'{patient.id} ({patient.gender}, {patient.age_group}'
                else:
                    room_patient[room][index][i] = patient.id
                color = Color(pick_for=patient.id, pick_key=None)
                color_map[patient.id] = (color.get_hex(), contrast_color(color).get_hex())

            if isinstance(patient, NewPatient):
                index = self.__find_index(surgeon_patient[patient.surgeon_id], admission, self.instance.days)
                if details:
                    surgeon_patient[patient.surgeon_id][index][admission] = (f'{patient.id} ('
                                                                             f'R: {patient.surgery_release_day}, '
                                                                             f'D: {patient.surgery_duration})')
                else:
                    surgeon_patient[patient.surgeon_id][index][admission] = patient.id

        # Unscheduled patients
        for p in sorted([p for p in self.patients.values() if p.admission_day is None], key=lambda x: x.id):
            patient = self.instance.patients[p.id]
            index = self.__find_index(unscheduled_patients, patient.surgery_release_day, self.instance.days)
            if details:
                unscheduled_patients[index][patient.surgery_release_day] = (f'{patient.id} ({patient.gender}, '
                                                                            f'{patient.age_group}, '
                                                                            f'D: {patient.surgery_duration})')
            else:
                unscheduled_patients[index][patient.surgery_release_day] = patient.id
            color = Color(pick_for=patient.id, pick_key=None)
            color_map[patient.id] = (color.get_hex(), contrast_color(color).get_hex())

        indices = [f'D{d}' for d in range(self.instance.days)]
        data_dict = {'shift': indices + ['room_group']}
        self.__extract_assignments(data_dict, room_patient, '{a}-{n}')
        self.__extract_assignments(data_dict, surgeon_patient, '{a}-{n}', format_first='{a}-Avail')
        for n, content in enumerate(unscheduled_patients):
            data_dict[f'u-{n}'] = content + ['unscheduled']

        frame = pd.DataFrame(data_dict).set_index('shift').T.rename_axis('room').reset_index()

        def map_color(content, index):
            if isinstance(content, str):
                content = content.split(' ')[0]
            if content in color_map:
                return color_map[content][index]
            return ('grey', 'black')[index]

        table = (GT(frame, rowname_col='room', groupname_col='room_group')
                 .tab_header(f'Solution for instance {self.instance.instance_file}')
                 .tab_stubhead('Rooms'))
        for day in indices:
            table = table.tab_style(style=[style.fill(color=lambda df: df[day].map(lambda x: map_color(x, 0))),
                                           style.text(color=lambda df: df[day].map(lambda x: map_color(x, 1)))],
                                    locations=loc.body(columns=day, rows=lambda df: ~df['room'].str.endswith('Avail')))
        table.show()


if __name__ == '__main__':
    instance = Instance.from_file(sys.argv[1])
    solution = Solution.from_file(instance, sys.argv[2])
    solution.print_table(len(sys.argv) > 3)
