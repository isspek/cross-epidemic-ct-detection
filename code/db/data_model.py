from dataclasses import dataclass


@dataclass
class PostInstance:
    post_id: str
    user_id: str
    external_link: str
    external_link_credibility: str
    created_at: str
    checkworthy_claim: float
