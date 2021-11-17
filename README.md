# Strudel
Detecting structure in verbose CSV files via classifying lines and cells.

## Getting Started

### Installing

* This project is implemented in Python 3.7.7.
* Use the following command to download all required libraries for Python:
```
pip install -r requirements.txt
```
* We recommend to install the required libraries in a separated virtual environment.

### Executing program

* Use the following script to run the Strudel program:
```
python run_strudel.py
```
The following arguments can be used for the above script:
* -d: training dataset
* -t: test dataset. If not given, the program does cross-validation on the training dataset
* -f: dataset path
* -o: output path

* Results are stored in a csv file.

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) License - see the LICENSE.md file for details

## Acknowledgments

* [Gerardo Vitagliano](https://github.com/vitaglianog)
* [Felix Naumann](https://github.com/felix-naumann)

## Contact
Please contact [Lan Jiang](mailto:lan.jiang@hpi.de) if you have any questions or want to report bugs.

## Reference

If you find this repository useful in your work, please cite our [EDBT'21 paper](https://edbt2021proceedings.github.io/docs/p32.pdf):

```
@inproceedings{jiang2021structure,
  title={Structure Detection in Verbose CSV Files.},
  author={Jiang, Lan and Vitagliano, Gerardo and Naumann, Felix},
  booktitle={EDBT},
  pages={193--204},
  year={2021}
}
```
