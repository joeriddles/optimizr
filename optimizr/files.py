from __future__ import annotations
import json
from typing import Optional

from .groups import Group


def read_groups(filename: str) -> list[Group]:
    groups: list[Group] = []
    with open(filename) as groups_json:
        groups_data: list[dict] = json.load(groups_json)
    for item in groups_data:
        try:
            group_members: list[str] = item['Members']
        except KeyError:
            pass
        else:
            group_id: Optional[str] = item.get('Group', None)
            group = Group(group_members, group_id)
            groups.append(group)
    return groups
