# Classical Mating Kernel System

A sophisticated PyQt5-based desktop application implementing the Computational Theory of AI Copulation and Hybridization. This system provides advanced evolutionary algorithms with features including lineage tracking, epigenetic systems, neural interfaces, and quantum entanglement simulation capabilities.

## Features

- **Advanced Mating Kernel**: 6-dimensional parameter space for evolutionary recombination
- **Lineage Tracking**: Cryptographic ancestry tracing with immutable lineage identifiers
- **Epigenetic System**: Reversible phenotypic switches with developmental plasticity
- **Neural Interface**: Social-genetic interactions and coordinated recombination
- **Quantum Simulation**: Quantum entanglement and non-local correlation effects
- **Topological Analysis**: Advanced diversity metrics and population structure analysis
- **Kill-Switch Protocol**: Ecological safeguards to prevent pathological evolution
- **Comprehensive Visualization**: Real-time metrics, population distribution, and lineage trees

## System Requirements

### Python Version
- **Python 3.8+** required

### Operating Systems
- **Linux** (Ubuntu, Debian, CentOS, etc.) - Fully supported
- **macOS** (10.15+) - Supported with Qt5 installation
- **Windows** (10+) - Supported but may require additional Qt5 setup

### Hardware
- **CPU**: Multi-core processor recommended
- **RAM**: Minimum 4GB, 8GB+ recommended for larger populations
- **Storage**: Minimal requirements (few MB for application)
- **Display**: Desktop environment with X11/Wayland support

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install qt5-default python3-pyqt5 python3-dev build-essential
```

**macOS:**
```bash
brew install qt5
```

**Windows:**
Download and install Qt5 from [qt.io](https://www.qt.io/download)

### 2. Install Python Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or install manually
pip install numpy pandas matplotlib scipy scikit-learn networkx PyQt5 markdown2
```

### 3. Run the Application

```bash
python3 main.py
```

## Usage

### Basic Operation

1. **Configure Parameters**: Adjust the 6-dimensional kernel parameters (α, β, γ, δ, ε, ζ)
2. **Set Population Parameters**: Configure population size, generations, genome dimension
3. **Enable Advanced Features**: Toggle lineage tracking, epigenetic system, neural interface
4. **Start Simulation**: Click "Start Simulation" to begin evolutionary process
5. **Monitor Progress**: View real-time metrics in the simulation tab
6. **Analyze Results**: Use visualization and analysis tabs to explore results

### Key Parameters

- **α (Symmetry)**: Balance of parental contributions (0.0-1.0)
- **β (Stochasticity)**: Mutation intensity (0.0-5.0)
- **γ (Granularity)**: Recombination resolution (gene/block/module)
- **δ (Compatibility)**: Minimum distance for hybridization (0.0-1.0)
- **ε (Fluke Contingency)**: Novelty injection rate (0.0-0.5)
- **ζ (Speciation Temperature)**: Selection pressure (0.0-5.0)

### Example Configurations

**Basic Evolutionary Optimization:**
```
α = 0.5, β = 1.0, γ = 0.3, δ = 0.7, ε = 0.05, ζ = 1.0
```

**Speciation Experiment:**
```
α = 0.7, β = 0.5, γ = 0.8, δ = 0.9, ε = 0.01, ζ = 0.3
```

**Innovation Search:**
```
α = 0.3, β = 2.0, γ = 0.2, δ = 0.5, ε = 0.1, ζ = 2.0
```

## Theoretical Foundations

The system implements formal theorems from the Computational Theory of AI Copulation:

1. **No-Free-Mating Theorem**: Universal copulation requires unbounded complexity
2. **Hybrid Vigor Bound**: Fitness gain bounded by parental variance
3. **Lineage Entropy Growth**: Guaranteed logarithmic emergence of novelty

Each agent is defined by the extended tuple: `A = 〈g, ι, τ, ρ, θ, σ〉`
- `g`: Genotype (weights/circuits/programs)
- `ι`: Immutable lineage identifier
- `τ`: Epigenetic tag (reversible phenotypic switch)
- `ρ`: Fitness scalar
- `θ`: Learned reward model parameters
- `σ`: Ecological safeguard meta-genes

## File Structure

```
.
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This documentation file
└── templates/
    ├── documentation.html  # Theoretical documentation
    └── index.html         # Web interface template
```

## Advanced Features

### Ecological Safeguards
- **Population Floor (N_min)**: Minimum population size (default: 20)
- **Archive Rate (λ₀)**: Lateral gene transfer rate (default: 0.15)
- **Epigenetic Refresh (T_pulse)**: Epigenetic reset period (default: 10 generations)
- **Kill-Switch Protocol**: Terminates lineages under pathological conditions

### Export Capabilities
- **JSON Export**: Complete simulation data with metadata
- **CSV Export**: Generation metrics and parameters
- **Excel Export**: Multi-sheet analysis (requires openpyxl)
- **Plot Export**: High-resolution visualizations (PNG/PDF/SVG)

## Development

### Adding New Features
1. **New Parameters**: Add UI controls in `setup_simulation_tab()`
2. **New Algorithms**: Extend `SimulationThread` methods
3. **New Visualizations**: Create canvas classes and add to appropriate tabs

### Code Structure
- **SimulationThread**: Core evolutionary algorithm implementation
- **ClassicalMatingKernelApp**: Main GUI application class
- **MplCanvas**: Matplotlib integration for plotting
- **PopulationVisualization**: Population distribution visualization

## Troubleshooting

### Common Issues

**Qt5 Import Errors:**
```bash
# Ensure Qt5 libraries are installed
sudo apt-get install qt5-default  # Ubuntu/Debian
brew install qt5                 # macOS
```

**Matplotlib Backend Issues:**
```python
# The application uses Qt5Agg backend
matplotlib.use('Qt5Agg')
```

**Memory Issues with Large Populations:**
- Reduce population size or genome dimension
- Increase kill-switch thresholds
- Enable population floor safeguards

### Performance Optimization
- Use smaller population sizes for initial testing
- Reduce genome dimension for faster computation
- Adjust simulation speed parameter
- Disable advanced features if not needed

## License

GNU General Public License v3.0

## Citation

If you use this software in research, please cite:

```
Mareya, L. A. (2025). Classical Mating Kernel: Computational Theory of AI Copulation and Hybridization.
```

## Support

For issues and questions:
1. Check the documentation in the application's "Documentation" tab
2. Review the theoretical foundations in `templates/documentation.html`
3. Examine example configurations for different use cases

## Contributing

Contributions to enhance the theoretical framework or implementation are welcome. Please ensure any changes maintain compatibility with the existing ecological safeguards and kill-switch protocols.
