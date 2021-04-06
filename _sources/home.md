# Protein 3D Poses Recovery

**3D Poses Recovery in Single-Particle Cryo-EM from Learned Pairwise Projection Distances**

The topic of this project is to learn pairwise projection distances in order to recover the angles at which we imaged these 2D projections from a given 3D protein.

## Summary
Single-particle cryo-electron microscopy (cryo-EM) is a technology that allows the observation and the high-resolution 3D structure determination of biomolecules. In this project, the goal is to estimate the angles at which we imaged the 2D projections from a given 3D protein (cf illustration bellow). We developed deep learning models to estimate the angles from learned pairwise projection distances. We designed a two-step method: 1) **distance estimation** using a Siamese neural network to learn the distance between pairs of projections, and 2) **orientation recovery** that includes a minimization scheme in order to estimate the angles at which each projection was taken. The current results obtained are discussed depending on different combination of approaches used andexperimental conditions.
![images/spcryoem.png](images/spcryoem.png)

## General Flow
General flow of the project can be seen in the illustration bellow:
![images/protein_flow.png](images/protein_flow.png)

## Team
**Student:**  
[Jelena Banjac](https://jelenabanjac.com), jelena.banjac@epfl.ch, Data Science Master Student

**Supervisors:**  
[Laurène Donati](https://people.epfl.ch/laurene.donati?lang=en), laurene.donati@epfl.ch, BIG, EPFL  
[Michaël Defferrard](https://deff.ch/), michael.defferrard@epfl.ch, LTS2, EPFL

**Professor:**  
[Michaël Unser](http://bigwww.epfl.ch/unser/), michael.unser@epfl.ch, BIG, EPFL

