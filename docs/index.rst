.. research documentation master file, created by
   sphinx-quickstart on Wed Mar 25 11:15:49 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Research Repository
===================

This is a centralized source code repository for collaborative work at Purdue Digital Twin Lab, inspired by the success of `google3 repository <https://dl.acm.org/doi/pdf/10.1145/2854146?trk=public_post_comment-text>`_ at Google. The ideology is that such a monolithic codebase can scale to dozens of projects for the parallel development with multiple developers. The benefits from maintaining the repository include

* Promote efficiency in collaborative development.
   - Maintaining a unified software versioning.
   - Improving the reproducibility of academic research.
   - Enabling extensive code sharing to facilitate collaboration across teams.
   - Simplifying dependency management, atomic changes, and large-scale refactoring.
   - Allowing flexible code ownership and visibility in management.
* Encourage passing down of knowledge, skills, and best practices.
* Establish a standardized workflow for future projects.

However, an effective maintenance of the repository would require a joint effort of all the lab members, to ensure code health and resolve potential redundancy in unnecessary dependencies. Thereby, we encourage the developers to file Research Proposals (REPs) before creating a pull request. See documentation below.

.. toctree::
   :maxdepth: 2
   :caption: Research Proposals

   reps/0-introduction
