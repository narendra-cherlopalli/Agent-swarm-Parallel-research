"""
swarm/agent.py — Swarm / decentralized peer agent pattern.

Domain: Transfer pricing comparable research — N research agents each
independently investigate a different candidate comparable company,
with NO central coordinator directing their work. Agents occasionally
exchange findings peer-to-peer: if one agent discovers that a candidate
shares a parent company with another candidate already being researched
by a peer, it broadcasts that finding so the peer can factor it in
(shared-parent comparables are a known TP red flag — using two
comparables under common control understates true arm's-length variance).

The defining trait versus hierarchical manager-worker: there is no
manager assigning work or aggregating results top-down. Each agent
operates on its own assigned candidate and only interacts with peers
through a shared message bus, asynchronously and optionally — a peer
message changes an agent's own analysis, but no agent can command
another agent to do anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PeerMessage:
    sender: str
    subject_company: str
    finding: str


@dataclass
class ComparableAssessment:
    agent_id: str
    company: str
    is_viable: bool
    rejection_reason: Optional[str]
    peer_flags_received: list[str] = field(default_factory=list)


class MessageBus:
    """
    Shared peer-to-peer message channel. No central coordinator reads or
    routes messages — agents post to it and other agents independently
    check it for anything relevant to their own assigned company.
    """

    def __init__(self) -> None:
        self.messages: list[PeerMessage] = []

    def post(self, sender: str, subject_company: str, finding: str) -> None:
        self.messages.append(PeerMessage(sender=sender, subject_company=subject_company, finding=finding))

    def messages_about(self, company: str) -> list[PeerMessage]:
        return [m for m in self.messages if m.subject_company == company]


class ComparableResearchAgent:
    """
    One swarm peer, independently researching ONE candidate comparable
    company. Posts a peer warning if it discovers a shared-parent
    relationship with another candidate; checks the bus for warnings
    about its OWN candidate before finalizing its assessment.
    """

    def __init__(self, agent_id: str, bus: MessageBus, ownership_data: dict, financial_data: dict) -> None:
        self.agent_id = agent_id
        self.bus = bus
        self.ownership_data = ownership_data
        self.financial_data = financial_data

    def broadcast_findings(self, company: str, other_candidates: list[str]) -> None:
        """
        Phase 1: independently check for shared-parent relationships with
        OTHER candidates and broadcast a warning if found. Does not yet
        form a conclusion about this agent's own candidate.
        """
        my_parent = self.ownership_data.get(company, {}).get("parent")
        if not my_parent:
            return
        for other in other_candidates:
            if other == company:
                continue
            other_parent = self.ownership_data.get(other, {}).get("parent")
            if other_parent and other_parent == my_parent:
                self.bus.post(
                    sender=self.agent_id,
                    subject_company=other,
                    finding=f"{other} shares parent company '{my_parent}' with {company} — "
                            f"common control may compromise independence as a comparable.",
                )

    def finalize(self, company: str) -> ComparableAssessment:
        """
        Phase 2: independently assess financial viability of this agent's
        own candidate, then check the shared bus for any peer warnings
        about it — by this point ALL agents have completed their
        broadcast phase, so this check is order-independent.
        """
        financials = self.financial_data.get(company, {})
        is_financially_viable = self._assess_financials(financials)

        peer_warnings = self.bus.messages_about(company)
        peer_flags = [m.finding for m in peer_warnings]

        if not is_financially_viable:
            return ComparableAssessment(
                agent_id=self.agent_id, company=company, is_viable=False,
                rejection_reason="Financial profile outside acceptable comparable range.",
                peer_flags_received=peer_flags,
            )

        if peer_warnings:
            return ComparableAssessment(
                agent_id=self.agent_id, company=company, is_viable=False,
                rejection_reason=f"Rejected based on peer finding: {peer_warnings[0].finding}",
                peer_flags_received=peer_flags,
            )

        return ComparableAssessment(
            agent_id=self.agent_id, company=company, is_viable=True,
            rejection_reason=None, peer_flags_received=peer_flags,
        )

    def research(self, company: str, other_candidates: list[str]) -> ComparableAssessment:
        """
        Convenience method for single-agent standalone use (runs both
        phases for ONE agent in isolation). NOT used by the swarm
        orchestrator — ComparableResearchSwarm.run() calls
        broadcast_findings() and finalize() separately across ALL agents
        to avoid the ordering dependency this combined method has when
        multiple agents run sequentially against a shared bus.
        """
        self.broadcast_findings(company, other_candidates)
        return self.finalize(company)

    @staticmethod
    def _assess_financials(financials: dict) -> bool:
        margin = financials.get("operating_margin_pct")
        if margin is None:
            return False
        return 1.0 <= margin <= 15.0  # plausible operating margin range for a comparable


class ComparableResearchSwarm:
    """
    Orchestrates running each peer's research() independently. This class
    exists only to fan out the work and collect results — it is NOT a
    manager in the hierarchical sense, since it makes no decisions about
    HOW each agent should research, doesn't retry failures, and doesn't
    aggregate via synthesis logic. It just runs N independent peers and
    returns their individually-formed conclusions.
    """

    def __init__(self, ownership_data: dict, financial_data: dict) -> None:
        self.ownership_data = ownership_data
        self.financial_data = financial_data

    def run(self, candidates: list[str]) -> list[ComparableAssessment]:
        bus = MessageBus()
        agents = [
            ComparableResearchAgent(
                agent_id=f"researcher_{i}", bus=bus,
                ownership_data=self.ownership_data, financial_data=self.financial_data,
            )
            for i, _ in enumerate(candidates)
        ]

        # ── Phase 1: every agent broadcasts shared-parent findings ──────────
        # This must complete for ALL agents before ANY agent finalizes —
        # otherwise the conclusion would depend on candidate ordering,
        # which defeats the purpose of decentralized peer discovery.
        for agent, company in zip(agents, candidates):
            agent.broadcast_findings(company, candidates)

        # ── Phase 2: every agent forms its final conclusion ──────────────────
        # By this point all peer broadcasts are visible on the bus regardless
        # of which agent discovered them or in what order.
        results = [
            agent.finalize(company)
            for agent, company in zip(agents, candidates)
        ]
        return results
