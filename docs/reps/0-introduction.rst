REP 0 --- Introduction to REP
=============================

**Author**: Juanwu Lu

**Status**: Active

**Type**: Informational

**Created**: 25-Mar-2026

.. contents:: Table of Contents

What is it?
-----------

A Research Proposal (REP) is a single design document that delivers information to the members of Purdue Digital Twin Lab, requesting new features or direction of interest to investigate. A valid REP should present a concise technical specification of the proposed direction or feature and rationale for its investigation.

My expectation for REPs is to promote a standardized workflow for collaborations and discussion, by collecting feedback from lab fellows on an interesting direction, and for documenting the design decisions that have been implemented in a future project. Therefore, the REP authors are responsible for establishing consensus within the collaborators and documenting dissenting opinions.

What are the types of REPs?
---------------------------

To serve different use cases of a proposal, there are three kinds of REPs:

1. A **Standard REP** proposes a new research direction for investigation and should generally be associated with one or multiple ongoing projects. It should contain detailed design and outline key milestones and timeline for the implementation.
2. A **Feature REP** describes a new feature, or proposes a change to the existing functionality in the repository. Feature REPs are like Standard REPs but focuses on proposing and implementing a functional feature that can serve multiple different projects and promote collaboration.
3. An **Informational REP** proposes and provides general guidelines or information to the lab members. However, it does not propose a new research direction for investigation and should not encourage implementation of any new features or changes.

How to start a REP?
-------------------
To start with, create an issue with an `REP label <https://github.com/PurdueDigitalTwin/research/issues?q=label%3AREP>`_. All consecutive pull requests that are associated with the REP (*e.g., new features or bug fixes*) should be linked to this root issue. Meanwhile, create a pull request that adds a file named after *%d-{short-title}.rst* under *docs/reps* with the first number being the issue number. In particular, this REP uses zero numbering to serve as a guideline for the consecutive ones.
