# Structural Banking

Structural banking is a framework to study banking mechanisms in a dynamic environment. The repository contains codes to solve and calibrate the equilibrium of a banking system in either standard policy or quantitative easing (QE). The calibration for QE is then used to study the counterfactual issuance of CBDC under different assumptions.

This project is the result of my time at the Bank of England as part of the PhD intern program. It is an extension of the work presented in the paper “CBDC and Banks: Threat or Opportunity?”, co-authored with Luciano Somoza.

## Authors
- [Martina Fraschini](https://www.martinafraschini.com/) (University of Luxembourg)
- [Luciano Somoza](https://www.lucianosomoza.com/) (University of Lausanne and Swiss Finance Institute)

## Documentation
The pdf document [documentation.pdf](https://github.com/MartinaFraschini/structural_banking/documentation.pdf) present the model in detail and the calibration strategy for standard policy and QE.

## Installation
The codes run on python 3.

All the packages you need:
- numpy
- pandas
- scipy
- matplotlib
- datetime

## Usage
The codes in the [parallel](https://github.com/MartinaFraschini/structural_banking/parallel) folder start in parallel for different values of the parameters. If you want to run a specific scenario on your laptop, you can use the codes in the [single](https://github.com/MartinaFraschini/structural_banking/single) folder.

At the moment there are 5 projects:

### 01_base_calib [[here](https://github.com/MartinaFraschini/structural_bankin/single/01_base_calib)]

It calibrates the baseline model under standard policy (Section 1 in the documentation) with UK data. The calibration part follows the strategy in Section 3 in the documentation.

### 02_qe_calib [[here](https://github.com/MartinaFraschini/structural_banking/single/02_qe_calib)]

It calibrates the model under quantitative easing (Section 2 in the documentation) with UK data. This part matches the equilibrium reached with QE in the years 2009-2020 and does not focus on the specific dynamics to reach the equilibrium. The calibration part follows the strategy in Section 3 in the documentation.

### qe_dynamics [[here](https://github.com/MartinaFraschini/structural_banking/parallel/qe_dynamics)]

It calibrates the model under quantitative easing (Section 2 in the documentation) with UK data. This part look at the dynamics used to implement QE in the years 2009-2020. _Not completely ready yet._

### 03_qe_cbdc_nofund [[here](https://github.com/MartinaFraschini/structural_banking/single/03_qe_cbdc_nofund)]

It take the calibration found with the code in [02_qe_calib](https://github.com/MartinaFraschini/structural_banking/single/02_qe_calib) and add a CBDC as a counterfactual (Section 4 in the documentation). In this part the commercial bank cannot ask the central bank to compensate the loss in deposits. Households can have different distributions in their preference for CBDC. The CBDC interest rate can take different values.

### 04_qe_cbdc_fund [[here](https://github.com/MartinaFraschini/structural_banking/single/03_qe_cbdc_fund)]

It take the calibration found with the code in [02_qe_calib](https://github.com/MartinaFraschini/structural_banking/single/02_qe_calib) and add a CBDC as a counterfactual (Section 4 in the documentation). In this part the commercial bank can ask the central bank to compensate the loss in deposits only when it cannot swap reserves into CBDC anymore. Households can have different distributions in their preference for CBDC. The CBDC interest rate and the interest rate for the central bank's funding can take different values.

## Running the code

Launch the code in the [single](https://github.com/MartinaFraschini/structural_banking/single) folder with:
```
$ <PATH to the repository>/structural_banking/single/<PATH to project>
$ python main.py
```

Launch the code in the [parallel](https://github.com/MartinaFraschini/structural_banking/parallel) folder with:
```
$ <PATH to the repository>/structural_banking/parallel/<PATH to project>
$ sbatch run_array.sh
```

To change the value of the parameter, always go to the _params.py_ file.
