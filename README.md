# fast_tffm
An optimized factorization machine implementation based on tensorflow

Working on benchmarks with other implementation and distributed version.

## Usage

### Compile
```
mkdir build
cd build
cmake ../
make
make test
cd ..
```

### Training
```
python fast_tffm.py train sample.cfg
```
### Prediction
```
python fast_tffm.py predict sample.cfg
```

