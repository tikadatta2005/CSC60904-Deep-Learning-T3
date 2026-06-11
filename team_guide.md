# Guide for teammates to update the repo code

## Experiments And Iterations
1. Each small change will be in a new file and treated as a new experiment 
2. Each Experiment must contain both Train and Val metrics comparison
3. Each Experiment must have a documentation in `/documentations/experiments`
4. Experiment must contain descriptive markdowns
5. Every Experiment must be examined before going to next Experiment
6. Discussion in Group is suggested before conducting any Experiment
7. Possible Experiments for the team can be:
    - 4 Conv Blocks without BatchNorm
    - 4 Conv Blocks with BatchNorm
    - Try 20 Conv Blocks with BatchNorm and Compare with 4 Conv
    - Try 20 Conv with ResNet and compare with Previous
    - Start minimum Learning Rate `0.001` or `1e-3`
    - Compare model on Adam and SGD Optimizer.
    - Remove, adjust MaxPool2D Layers
    - Experiment with Multiple Classification Layers
8. Disuss Architecture and Module Add And Change with Team Before trying yourself
9. Always Use the Common Modules no Additional Unknown modules