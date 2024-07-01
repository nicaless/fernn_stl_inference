# README

Code for the paper "Differentiable Inference of Temporal Logic Formulas" presented in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems

Paper Abstract:
We demonstrate the first recurrent neural network architecture for learning signal temporal logic (TL) formulas, and present the first systematic comparison of formula inference methods. Legacy systems embed much expert knowledge which is not explicitly formalized. There is great interest in learning formal specifications that characterize the ideal behavior of such systems—that is, formulas in TL that are satisfied by the system’s output signals. Such specifications can be used to better understand the system’s behavior and improve the design of its next iteration. Previous inference methods either assumed certain formula templates, or did a heuristic enumeration of all possible templates. This work proposes a neural network architecture that infers the formula structure via gradient descent, eliminating the need for imposing any specific templates. It combines the learning of formula structure and parameters in one optimization. Through systematic comparison, we demonstrate that this method achieves similar or better misclassification rates (MCRs) than enumerative and lattice methods. We also observe that different formulas can achieve similar MCR, empirically demonstrating the under-determinism of the problem of TL inference.

Citation:
```
@ARTICLE{9852801,
  author={Fronda, Nicole and Abbas, Houssam},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={Differentiable Inference of Temporal Logic Formulas}, 
  year={2022},
  volume={41},
  number={11},
  pages={4193-4204},
  keywords={Lattices;Behavioral sciences;Semantics;Training;Systematics;Recurrent neural networks;Computer architecture;Formal methods;inference;recurrent neural networks (RNNs);temporal logic (TL)},
  doi={10.1109/TCAD.2022.3197506}}
```

## Main Files

* `TLOps.py` - Implementations for STL operators
* `TL_learning_utils.py` - auxiliary layers for building STL formulae
* `/experiments` - folder containing experiments using STL RNN architecture, including those presented in the paper.
