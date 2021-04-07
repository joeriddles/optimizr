from typing import Optional


class Group:
    group_id: Optional[str]
    members: list[str]

    def __init__(self, members: list[str], group_id: Optional[str]=None):
        self.members = members
        self.group_id = group_id
