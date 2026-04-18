# aria/user_profile.py
# Cross-session persistent memory using local JSON user profiles.

import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional

PROFILES_DIR = Path(__file__).parent.parent / "data" / "profiles"


def load_profile(user_id: str) -> dict:
    """Load user profile from JSON file, or return empty profile if not found."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{user_id}.json"
    if profile_path.exists():
        with open(profile_path, "r") as f:
            return json.load(f)
    return {
        "user_id": user_id,
        "session_count": 0,
        "topics_researched": [],
        "last_active": None,
        "preferred_domains": [],
    }


def save_profile(user_id: str, profile: dict) -> None:
    """Write user profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{user_id}.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)


def update_profile(profile: dict, question: str, report: dict, sub_queries: Optional[List[str]] = None) -> dict:
    """
    Update user profile with data from the latest completed research turn.

    Extracts topics from sub_queries and infers domain from report summary.
    """
    profile["session_count"] = profile.get("session_count", 0) + 1
    profile["last_active"] = datetime.now().isoformat()

    # Add sub-query topics (deduplicated)
    existing_topics = set(profile.get("topics_researched", []))
    if sub_queries:
        for q in sub_queries[:5]:
            existing_topics.add(q.strip()[:80])
    profile["topics_researched"] = list(existing_topics)[-50:]  # cap at 50

    # Infer domain from summary
    if isinstance(report, dict):
        summary = report.get("summary", "")
        if summary:
            # Simple domain inference from first 3 words
            words = summary.split()[:5]
            domain_hint = " ".join(words)
            domains = set(profile.get("preferred_domains", []))
            domains.add(domain_hint[:40])
            profile["preferred_domains"] = list(domains)[-10:]  # cap at 10

    return profile


def build_user_context(profile: dict) -> str:
    """Build user context string for injection into LLM prompts."""
    topics = profile.get("topics_researched", [])
    domains = profile.get("preferred_domains", [])
    sessions = profile.get("session_count", 0)

    if not topics and not domains:
        return ""

    parts = []
    if topics:
        parts.append(f"This researcher has previously studied: {', '.join(topics[-10:])}")
    if domains:
        parts.append(f"Preferred domains: {', '.join(domains[-5:])}")
    if sessions > 1:
        parts.append(f"They have completed {sessions} research sessions.")

    return ". ".join(parts)
