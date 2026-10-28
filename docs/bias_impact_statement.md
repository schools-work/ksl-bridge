# Bias impact statement (adapted from Brookings)

> Update for every major release. Owner: the research/community lead. Reviewed by Deaf partners.

1. **What does the system produce, and who is affected?**
   Sign → text/speech for hearing people and machines; text/speech → signing for KSL users. Errors affect Deaf and hard-of-hearing users most.
2. **Which groups could be harmed, and how severely?**
   * Signers under-represented in training: older signers, rural or regional variants, left-handed signers, darker skin in poor lighting (landmark-detector measurement bias).
   * Severity depends on the domain: a wrong drink is mild; a wrong allergy or medication is severe (critical glosses, VOI severity ×4 in hospitals).
3. **Whose signing does the data represent, and whose doesn't it?**
   See `dataset.v_build_composition` and the dataset cards. List missing regions explicitly.
4. **Per-group results:** `ml.v_fairness_gaps` for the production models (paste the table here each release).
5. **Mitigations taken:** recruitment quotas, variant glosses kept, mild bias cap, clarification instead of silent correction, personalisation, low-light data.
6. **Contest and correct:** the ✏️ correction on every utterance (`feedback.events`), human escalation (`agent.escalations`), and a feedback channel to the team.
7. **Review:** before each promotion to production (`ml.release_gates.gate = 'fairness'`).
