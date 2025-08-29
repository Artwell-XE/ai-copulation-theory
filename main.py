# Computational Theory of AI Copulation and Hybridization
# Classical Mating Kernel Simulation
# By Liberty A. Mareya, 2025

from collections import deque
import sys
import numpy as np
import pandas as pd
import matplotlib
import hashlib
import json
import networkx as nx
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
                             QGroupBox, QDoubleSpinBox, QSizePolicy, QFileDialog, QStatusBar, QSplashScreen,
                             QFrame, QSpinBox, QProgressBar, QMessageBox, QCheckBox, QLineEdit, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QLinearGradient, QBrush, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import random
import time
from datetime import datetime
import webbrowser
import markdown2

class SplashScreen(QSplashScreen):
    def __init__(self, pixmap=None):
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(800, 500)
        
        # Center splash screen
        screen_geometry = QApplication.primaryScreen().geometry()
        self.move(
            (screen_geometry.width() - self.width()) // 2,
            (screen_geometry.height() - self.height()) // 2
        )
        
        # Create progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(150, 420, 500, 25)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 10px;
                background-color: #2c3e50;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 8px;
            }
        """)
        
        # Create progress label
        self.progress_label = QLabel("Initializing...", self)
        self.progress_label.setGeometry(150, 390, 500, 25)
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("color: #ecf0f1; font-size: 12px;")
        
        # Animation variables
        self.helix_angle = 0
        self.helix_points = []
        self.progress_value = 0
        self.messages = [
            "Loading core modules...",
            "Initializing mating kernels...",
            "Preparing simulation environment...",
            "Building UI components...",
            "Configuring lineage tracking...",
            "Setting up epigenetic system...",
            "Initializing kill-switch protocol...",
            "Calibrating quantum entanglement...",
            "Establishing neural interfaces...",
            "Starting application..."
        ]
        
        # Start animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(30)
    
    def animate(self):
        # Update progress
        if self.progress_value < 100:
            self.progress_value += 0.5  # Slower progress
            self.progress_bar.setValue(int(self.progress_value))
            
            # Update message every 10%
            if self.progress_value % 10 == 0:
                msg_idx = min(int(self.progress_value // 10), len(self.messages) - 1)
                self.progress_label.setText(self.messages[msg_idx])
        
        # Update helix animation
        self.helix_angle += 0.1
        self.repaint()
    
    def paintEvent(self, event):
        # Draw background
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#1a1a2e"))
        gradient.setColorAt(1, QColor("#16213e"))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Draw decorative DNA background
        painter.setPen(QPen(QColor("#1e3a5f"), 1))
        for i in range(0, self.width(), 20):
            painter.drawLine(i, 0, i, self.height())
        
        # Draw title
        painter.setFont(QFont("Arial", 32, QFont.Bold))
        painter.setPen(QPen(QColor("#ecf0f1")))
        painter.drawText(QRectF(0, 120, self.width(), 50), Qt.AlignCenter, "CLASSICAL MATING KERNEL")
        
        # Draw subtitle
        painter.setFont(QFont("Arial", 14))
        painter.setPen(QPen(QColor("#bdc3c7")))
        painter.drawText(QRectF(0, 170, self.width(), 30), Qt.AlignCenter, 
                         "Computational Theory of AI Copulation and Hybridization")
        
        # Draw copyright
        painter.setFont(QFont("Arial", 10))
        painter.setPen(QPen(QColor("#7f8c8d")))
        painter.drawText(QRectF(0, 460, self.width(), 20), Qt.AlignCenter, 
                         "© Computational Theory of AI Copulation")
        
        # Draw DNA helix
        painter.setPen(QPen(QColor("#3498db"), 3))
        center_x = self.width() / 2
        center_y = 300
        
        # Draw connecting lines
        for i in range(0, 360, 15):
            rad = (i + self.helix_angle) * np.pi / 180
            x1 = center_x + 150 * np.cos(rad)
            y1 = center_y + 50 * np.sin(rad)
            x2 = center_x + 180 * np.cos(rad)
            y2 = center_y + 50 * np.sin(rad)
            
            # Calculate color based on position
            color_intensity = int(128 + 127 * np.sin(rad))
            color = QColor(color_intensity, color_intensity, 255)
            painter.setPen(QPen(color, 3))
            
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


class SimulationThread(QThread):
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal()
    kill_switch_activated = pyqtSignal()
    population_snapshot = pyqtSignal(object)
    
    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params
        self.paused = False
        self.stopped = False
        self.lineage_hashes = {}
        self.epigenetic_tags = {}
        self.lineage_tree = defaultdict(list)
        self.developmental_plasticity = params.get("developmental_plasticity", 0.1)
        self.current_generation = 0
        self.kill_switch_triggered = False
        self.population_history = []
        self.neural_interface = params.get("neural_interface", False)
        self.topological_features = params.get("topological_features", False)
        self.quantum_entanglement = False  # Quantum features disabled by default
        
        # New parameters for conceptual upgrades
        self.N_min = params.get("population_floor", 20)       # floor size
        self.lambda0 = params.get("archive_rate", 0.15)       # LGT rate
        self.T_pulse = params.get("epi_refresh_period", 10)   # τ refresh period
        self.sigma_ref = params.get("sigma_ref", 1.0)         # ζ adaptive ref
        self.tau_epsilon = params.get("tau_epsilon", 2.0)     # ε gate sensitivity
        self.archive = deque(maxlen=100)                      # extinct elites
        self.rolling_variance = deque(maxlen=5)               # for ζ_t
        
        # Seed RNG for deterministic reproducibility
        param_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        seed = int(param_hash[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
    
    def run(self):
        try:
            # Initialize population (ensure integer sizes)
            self.population = self.initialize_population(
                int(self.params["pop_size"]), 
                int(self.params["genome_dim"])
            )
            population = self.population.copy()
            
            # Initialize lineage tracking
            self.initialize_lineage_tracking(len(population))
            
            # Initialize epigenetic system
            self.initialize_epigenetic_system(len(population), int(self.params["genome_dim"]))
            
            # Main simulation loop (ensure integer generations)
            for gen in range(int(self.params["generations"])):
                self.current_generation = gen
                
                while self.paused and not self.stopped:
                    time.sleep(0.1)  # Wait while paused
                    
                if self.stopped or self.kill_switch_triggered:
                    break
                
                # Store population snapshot every 10 generations
                if gen % 10 == 0:
                    self.population_history.append(population.copy())
                
                # Apply developmental plasticity
                population = self.apply_developmental_plasticity(population)
                
                # Evaluate fitness - ensure it's always an array
                fitness = np.atleast_1d(self.evaluate_fitness(population))
                avg_fitness = np.mean(fitness)
                best_fitness = np.max(fitness)
                
                # Calculate diversity metrics
                diversity = self.calculate_diversity(population)
                topological_diversity = self.calculate_topological_diversity(population) if self.topological_features else 0

                # Lateral Archive Injection (after fitness evaluation)
                # Calculate lineage entropy H_t
                unique, counts = np.unique(list(self.lineage_hashes.values()), return_counts=True)
                H_t = -np.sum(counts * np.log(counts + 1e-12))
                archive_rate_t = self.lambda0 * max(0.0, 1.0 - H_t / max(H_t + 1, 1))
                if random.random() < archive_rate_t and self.archive:
                    idx = np.random.randint(0, len(population))
                    population[idx] = random.choice(self.archive)

                # Adaptive ζ_t (before parent selection)
                sigma_t = np.var(fitness) if len(fitness) > 1 else 1.0
                sigma_t = max(sigma_t, 1e-6)      # avoid zero
                ζ_t = 0.2 + (5.0 - 0.2) * np.exp(-sigma_t / self.sigma_ref)

                # Fitness-gated ε_t (same place)
                grad = np.mean(np.diff(fitness[-5:])) if len(fitness) >= 5 else 0.0
                ε_t = 0.15 * (1.0 / (1.0 + np.exp(self.tau_epsilon * grad)))  # Increased base ε from 0.05 to 0.15

                # Epigenetic Refresh Pulse
                if gen % self.T_pulse == 0:
                    for k in self.epigenetic_tags:
                        self.epigenetic_tags[k] = np.random.choice([0, 1], size=len(self.epigenetic_tags[k]))

                # Check kill-switch conditions
                if self.check_kill_switch_conditions(population, fitness):
                    self.kill_switch_activated.emit()
                    self.kill_switch_triggered = True
                    break
                
                # Select parents with neural interface if enabled
                parents = self.select_parents(population, fitness, ζ_t)  # Use adaptive ζ_t
                
                # Create new generation
                new_population = []
                new_lineage_hashes = {}
                new_epigenetic_tags = {}
                
                # Ensure we have valid parents
                if parents.size == 0:  # Explicit check for empty numpy array
                    parents = list(range(len(population)))  # Fallback to all individuals
                parents = [p for p in parents if 0 <= p < len(population)]  # Ensure valid indices
                
                for i in range(0, len(parents), 2):
                    if i+1 >= len(parents):
                        break
                    
                    # Get parents with bounds checking
                    parent1_idx = parents[i % len(parents)]  # Wrap around if needed
                    parent2_idx = parents[(i+1) % len(parents)]  # Wrap around if needed
                    
                    # Ensure indices are valid
                    if not (0 <= parent1_idx < len(population)) or not (0 <= parent2_idx < len(population)):
                        # Fallback to random valid parents
                        parent1_idx = random.randint(0, len(population)-1)
                        parent2_idx = random.randint(0, len(population)-1)
                    
                    parent1 = population[parent1_idx]
                    parent2 = population[parent2_idx]
                    
                    # Ensure lineage hashes exist for parents
                    if parent1_idx not in self.lineage_hashes:
                        self.lineage_hashes[parent1_idx] = self.generate_lineage_hash()
                    if parent2_idx not in self.lineage_hashes:
                        self.lineage_hashes[parent2_idx] = self.generate_lineage_hash()
                    
                    # Create two children - one symmetric, one asymmetric
                    child1 = self.mating_kernel(
                        parent1, parent2,
                        self.params["α"], self.params["β"], self.params["γ"],
                        self.params["δ"], ε_t
                    )
                    child2 = self.mating_kernel(
                        parent2, parent1,  # Reverse parent order for asymmetry
                        self.params["α"], self.params["β"], self.params["γ"],
                        self.params["δ"], ε_t
                    )
                    
                    # Apply mutation
                    child1 = self.apply_mutation(child1, self.params["mutation_rate"])
                    child2 = self.apply_mutation(child2, self.params["mutation_rate"])
                    
                    # Generate lineage hashes for both children
                    child1_hash = self.generate_child_lineage_hash(
                        self.lineage_hashes[parent1_idx],
                        self.lineage_hashes[parent2_idx]
                    )
                    child2_hash = self.generate_child_lineage_hash(
                        self.lineage_hashes[parent2_idx],
                        self.lineage_hashes[parent1_idx]
                    )
                    
                    # Apply epigenetic inheritance
                    child1_epigenetic = self.inherit_epigenetic_tags(
                        self.epigenetic_tags[parent1_idx],
                        self.epigenetic_tags[parent2_idx]
                    )
                    child2_epigenetic = self.inherit_epigenetic_tags(
                        self.epigenetic_tags[parent2_idx],
                        self.epigenetic_tags[parent1_idx]
                    )
                    
                    # Add both children to new population
                    new_population.extend([child1, child2])
                    new_lineage_hashes[len(new_population)-2] = child1_hash
                    new_lineage_hashes[len(new_population)-1] = child2_hash
                    new_epigenetic_tags[len(new_population)-2] = child1_epigenetic
                    new_epigenetic_tags[len(new_population)-1] = child2_epigenetic
                    
                    # Update lineage tree for both children
                    self.lineage_tree[child1_hash].extend([
                        self.lineage_hashes[parent1_idx],
                        self.lineage_hashes[parent2_idx]
                    ])
                    self.lineage_tree[child2_hash].extend([
                        self.lineage_hashes[parent2_idx],
                        self.lineage_hashes[parent1_idx]
                    ])
                
                # Ensure population size remains constant (convert to integer)
                new_size = int(self.params["pop_size"])
                self.population = np.array(new_population[:new_size])
                population = self.population.copy()
                
                # Ensure tracking structures match population size
                self.lineage_hashes = {i: new_lineage_hashes.get(i, self.generate_lineage_hash()) 
                                      for i in range(new_size)}
                self.epigenetic_tags = {i: new_epigenetic_tags.get(i, np.random.choice([0,1], size=self.params["genome_dim"]))
                                      for i in range(new_size)}

                # Population Floor & Soft-Restart - Enhanced with robust error handling
                if len(population) < self.N_min:
                    print(f"\n=== Population floor triggered ===")
                    print(f"Current size: {len(population)}, Min size: {self.N_min}")
                    
                    # Ensure we have at least one individual to clone from
                    if len(population) == 0:
                        print("CRITICAL: Empty population - creating new random population")
                        population = np.random.uniform(-10, 10, size=(self.N_min, self.params["genome_dim"]))
                        print(f"Created new population of size: {len(population)}")
                        continue  # Skip rest of floor logic since we have full population
                    
                    # Get valid top individuals with multiple fallback mechanisms
                    num_top = min(5, len(population))  # Never more than population size
                    print(f"Selecting top {num_top} individuals")
                    
                    # Multiple fallback mechanisms for getting top indices
                    top_indices = []
                    attempts = 0
                    max_attempts = 3
                    
                    while not top_indices and attempts < max_attempts:
                        try:
                            # Attempt 1: Use fitness-based selection
                            if len(fitness) == len(population):
                                fitness_arr = np.array(fitness) if not isinstance(fitness, np.ndarray) else fitness
                                top_indices = np.argsort(fitness_arr)[-num_top:]
                                top_indices = [i for i in top_indices if 0 <= i < len(population)]
                            
                            # Attempt 2: If fitness selection failed, use random selection
                            if not top_indices:
                                top_indices = np.random.choice(len(population), size=min(num_top, len(population)), replace=False)
                            
                            # Final validation
                            top_indices = [i for i in top_indices if 0 <= i < len(population)]
                            
                        except Exception as e:
                            print(f"Attempt {attempts+1} failed: {str(e)}")
                            top_indices = []
                        
                        attempts += 1
                    
                    # Ultimate fallback if all attempts fail
                    if not top_indices:
                        print("EMERGENCY: Could not select valid individuals - using first individual")
                        top_indices = [0] if len(population) > 0 else []
                    
                    # Create clones with multiple safety checks
                    clones = []
                    for i in top_indices:
                        try:
                            if 0 <= i < len(population):
                                # Create mutation with bounds checking
                                mutation = np.clip(
                                    np.random.normal(0, 0.01, population[i].shape),
                                    -0.1, 0.1  # Constrain mutation range
                                )
                                clones.append(np.clip(
                                    population[i] + mutation,
                                    -10, 10  # Keep within bounds
                                ))
                                print(f"Created clone from individual {i}")
                            else:
                                raise IndexError(f"Invalid index {i} for population size {len(population)}")
                        except Exception as e:
                            print(f"ERROR creating clone: {str(e)} - generating safe random individual")
                            clones.append(np.random.uniform(-10, 10, size=self.params["genome_dim"]))
                    
                    # Ensure we have at least one clone with multiple fallbacks
                    if not clones:
                        print("WARNING: No clones created - trying alternative methods")
                        
                        # Fallback 1: Use existing population if any
                        if len(population) > 0:
                            clones = [population[0] + np.random.normal(0, 0.01, population[0].shape)]
                        
                        # Fallback 2: Create completely new individuals
                        if not clones:
                            clones = [np.random.uniform(-10, 10, size=self.params["genome_dim"]) 
                                      for _ in range(num_top)]
                    
                    # Rebuild population with progress tracking and safety
                    new_population = list(population.copy())  # Convert to list for append operations
                    new_lineage_hashes = self.lineage_hashes.copy()
                    new_epigenetic_tags = self.epigenetic_tags.copy()
                    
                    target_size = max(self.N_min, len(population))
                    print(f"Rebuilding population from {len(new_population)} to {target_size}")
                    
                    added_count = 0
                    while len(new_population) < target_size and clones:
                        clone = clones.pop(0)
                        new_population.append(clone)
                        
                        # Create new lineage hash for clone
                        clone_idx = len(new_population) - 1
                        new_lineage_hashes[clone_idx] = self.generate_child_lineage_hash(
                            new_lineage_hashes[top_indices[0]],  # Use first parent's hash
                            self.generate_lineage_hash()         # New random component
                        )
                        
                        # Create new epigenetic tags for clone
                        if top_indices[0] in new_epigenetic_tags:
                            new_epigenetic_tags[clone_idx] = self.inherit_epigenetic_tags(
                                new_epigenetic_tags[top_indices[0]],
                                np.random.choice([0,1], size=len(new_epigenetic_tags[top_indices[0]]))
                            )
                        else:
                            new_epigenetic_tags[clone_idx] = np.random.choice([0,1], size=self.params["genome_dim"])
                        
                        added_count += 1
                        if not clones:  # Recycle clones if needed
                            clones = [c.copy() for c in new_population[-added_count:]]
                    
                    # Convert back to numpy array before continuing
                    population = np.array(new_population)
                    print(f"=== Population restored to size: {len(population)} ===")
                    continue  # Skip rest of generation logic since we just rebuilt

                # Emit generation data
                gen_data = {
                    "generation": gen,
                    "avg_fitness": avg_fitness,
                    "best_fitness": best_fitness,
                    "diversity": diversity,
                    "topological_diversity": topological_diversity,
                    "hybrid_vigor": self.calculate_hybrid_vigor(parents, fitness),
                    "fluke_rate": ε_t,  # Use current adaptive ε_t
                    "entropy": self.calculate_entropy(population),
                    "epigenetic_variation": self.calculate_epigenetic_variation(),
                    "quantum_entanglement": self.calculate_quantum_entanglement(),
                    "lineage_depth": self.calculate_lineage_depth(),
                    "adaptive_zeta": ζ_t,
                    "adaptive_lambda": archive_rate_t
                }
                self.update_signal.emit(gen_data)
                
                if gen % 5 == 0:  # Don't emit too frequently for performance
                    self.population_snapshot.emit(population)
                    
                    # Slow down simulation for visualization
                    time.sleep(self.params["speed"])
            
            # Finalize simulation
            self.finished_signal.emit()
        
        except Exception as e:
            print(f"Simulation error: {str(e)}")
            raise
    
    def initialize_population(self, pop_size, genome_dim):
        """Initialize random population with optional neural interface initialization"""
        if self.neural_interface:
            # Initialize with more structured patterns for neural interface
            base = np.random.uniform(-1, 1, size=genome_dim)
            self.population = np.array([base + np.random.normal(0, 0.5, size=genome_dim) for _ in range(pop_size)])
        else:
            self.population = np.random.uniform(-10, 10, size=(pop_size, genome_dim))
        return self.population.copy()
    
    def initialize_lineage_tracking(self, pop_size):
        """Initialize lineage tracking with cryptographic hashes"""
        self.lineage_hashes = {i: self.generate_lineage_hash() for i in range(pop_size)}
        self.lineage_tree = defaultdict(list)
    
    def initialize_epigenetic_system(self, pop_size, genome_dim):
        """Initialize epigenetic tagging system with optional neural patterns"""
        if self.neural_interface:
            # Create correlated epigenetic patterns
            base_pattern = np.random.choice([0,1], size=genome_dim)
            self.epigenetic_tags = {
                i: np.where(np.random.random(size=genome_dim) < 0.8, base_pattern, 1-base_pattern)
                for i in range(pop_size)
            }
        else:
            self.epigenetic_tags = {i: np.random.choice([0,1], size=genome_dim) for i in range(pop_size)}
    
    def generate_lineage_hash(self, data=None):
        """Generate a cryptographic lineage hash with timestamp"""
        if data is None:
            timestamp = str(datetime.now().timestamp()).encode()
            random_data = str(random.getrandbits(256)).encode()
            data = timestamp + random_data
        return hashlib.sha3_256(data).hexdigest()
    
    def generate_child_lineage_hash(self, parent1_hash, parent2_hash):
        """Generate a child's lineage hash from parent hashes with generation info"""
        combined = (parent1_hash + parent2_hash + str(self.current_generation)).encode()
        return self.generate_lineage_hash(combined)
    
    def evaluate_fitness(self, population):
        """Calculate fitness with positive rewards and neural enhancement"""
        if isinstance(population, np.ndarray):
            if len(population.shape) == 1:
                # Shift to positive rewards: 10 - sum(squares)
                fitness = 10.0 - np.sum(population**2)
            else:
                fitness = 10.0 - np.sum(population**2, axis=1)
            
            if self.neural_interface:
                # Add neural interface fitness component
                neural_component = np.abs(np.mean(population, axis=-1))  # Favor balanced weights
                fitness += neural_component * 0.5
            return np.maximum(fitness, 0.1)  # Ensure minimum fitness of 0.1
        return 0.1  # Default minimum fitness
    
    def calculate_diversity(self, population):
        """Calculate population diversity using multiple metrics"""
        if len(population) < 2:
            return 0.0
        
        # Genetic diversity (average pairwise distance)
        distances = pdist(population, 'euclidean')
        genetic_diversity = np.mean(distances) if len(distances) > 0 else 0.0
        
        # Phenotypic diversity (variance in fitness)
        fitness = self.evaluate_fitness(population)
        phenotypic_diversity = np.var(fitness)
        
        # Combined diversity metric
        return 0.7 * genetic_diversity + 0.3 * phenotypic_diversity
    
    def calculate_topological_diversity(self, population):
        """Calculate topological features of the population"""
        if len(population) < 5:  # Need enough points for meaningful topology
            return 0.0
        
        # Use PCA to reduce dimensionality
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(population)
        
        # Calculate alpha shape (convex hull area as proxy)
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(reduced)
            return hull.volume  # Area in 2D
        except:
            return 0.0
    
    def calculate_entropy(self, population):
        """Calculate entropy of population with multiple gene consideration"""
        if len(population) == 0:
            return 0.0
        
        # Calculate entropy across all genes
        entropies = []
        for gene_idx in range(population.shape[1]):
            gene_values = population[:, gene_idx]
            hist, _ = np.histogram(gene_values, bins=10, range=(-10, 10))
            prob = hist / np.sum(hist)
            prob = prob[prob > 0]  # Avoid log(0)
            entropies.append(-np.sum(prob * np.log(prob)))
        
        return np.mean(entropies)
    
    def calculate_lineage_depth(self):
        """Calculate lineage depth with caching for performance"""
        if not self.lineage_tree or not self.lineage_hashes:
            return 0
        
        # Cache depths to avoid recomputation
        depth_cache = {}
        
        def get_depth(hash_val):
            if hash_val in depth_cache:
                return depth_cache[hash_val]
            
            if not self.lineage_tree.get(hash_val, []):
                depth_cache[hash_val] = 1
                return 1
            
            depth = 1 + max(get_depth(p) for p in self.lineage_tree[hash_val])
            depth_cache[hash_val] = depth
            return depth
        
        try:
            return max(get_depth(h) for h in self.lineage_hashes.values())
        except ValueError:  # max() arg is empty sequence
            return 0
    
    def calculate_epigenetic_variation(self):
        """Calculate epigenetic variation with spatial patterns"""
        if not self.epigenetic_tags:
            return 0.0
        
        # Calculate spatial correlation of epigenetic tags
        tags_matrix = np.array(list(self.epigenetic_tags.values()))
        mean_tags = np.mean(tags_matrix, axis=0)
        spatial_correlation = np.mean(np.abs(np.diff(mean_tags)))
        
        # Combined variation metric
        active_tags = np.sum(tags_matrix)
        total_possible = tags_matrix.size
        activation_rate = active_tags / total_possible
        
        return 0.7 * activation_rate + 0.3 * spatial_correlation
    
    def calculate_quantum_entanglement(self):
        """Calculate degree of quantum entanglement in population"""
        if not hasattr(self, 'quantum_states') or len(self.quantum_states) < 2:
            return 0.0
        
        # Measure entanglement as correlation between quantum states
        correlation_matrix = np.corrcoef(self.quantum_states)
        if correlation_matrix.ndim >= 2:  # Only fill diagonal for 2D+ arrays
            np.fill_diagonal(correlation_matrix, 0)  # Ignore self-correlation
        return np.mean(np.abs(correlation_matrix))
    
    def select_parents(self, population, fitness, speciation_temp):
        """Select parents with optional neural interface enhancement"""
        # Handle cases where all fitness values are zero or invalid
        if np.all(fitness <= 0) or np.any(np.isnan(fitness)):
            # Fall back to uniform selection
            return np.random.choice(len(population), size=len(population), replace=True)
            
        # Apply softmax with temperature
        exp_fitness = np.exp(fitness / speciation_temp)
        probs = exp_fitness / np.sum(exp_fitness)
        
        # Handle any remaining NaN values
        if np.any(np.isnan(probs)):
            probs = np.ones(len(population)) / len(population)
        
        if self.neural_interface:
            # Neural interface adds social component to selection
            social_component = np.ones(len(population)) / len(population)  # Start with uniform
            for i in range(len(population)):
                # Social component based on similarity to others
                similarities = 1 / (1 + np.linalg.norm(population - population[i], axis=1))
                social_component[i] = np.mean(similarities)
            
            # Combine fitness and social components
            combined_probs = 0.7 * probs + 0.3 * (social_component / np.sum(social_component))
            return np.random.choice(len(population), size=len(population), p=combined_probs, replace=True)
        
        return np.random.choice(len(population), size=len(population), p=probs, replace=True)
    
    def mating_kernel(self, parent1, parent2, alpha, beta, gamma, delta, epsilon):
        """Enhanced mating kernel with quantum and neural features"""
        # Check compatibility
        distance = np.linalg.norm(parent1 - parent2)
        if distance < delta:
            # Not compatible for hybridization - return average
            return (parent1 + parent2) / 2.0
        
        # Apply fluke contingency (stochastic perturbation)
        if random.random() < epsilon:
            fluke_strength = random.uniform(0.5, 2.0)
            perturbation = np.random.random(size=parent1.shape) - 0.5
            if self.quantum_entanglement:
                # Quantum-enhanced perturbation
                perturbation *= np.random.normal(0, 1, size=parent1.shape)
            return parent1 + fluke_strength * perturbation
        
        # Neural interface adds coordinated recombination
        if self.neural_interface and random.random() < 0.3:
            # Coordinated block recombination based on neural patterns
            block_size = max(1, int(len(parent1) * 0.3))
            num_blocks = max(2, int(len(parent1) / block_size))
            blocks = [slice(i*block_size, (i+1)*block_size) for i in range(num_blocks)]
            mask = np.zeros(len(parent1))
            for block in blocks:
                mask[block] = np.random.randint(0, 2)
            return np.where(mask, parent1, parent2)
        
        # Standard recombination logic
        if gamma < 0.3:
            # Gene-level recombination
            mask = np.random.randint(0, 2, size=parent1.shape)
        elif gamma < 0.7:
            # Block-level recombination
            block_size = max(1, int(len(parent1) * 0.2))
            mask = np.zeros(len(parent1))
            for i in range(0, len(parent1), block_size):
                mask[i:i+block_size] = np.random.randint(0, 2)
        else:
            # Module-level recombination (single crossover point)
            crossover_point = random.randint(1, len(parent1)-1)
            mask = np.concatenate([np.ones(crossover_point), np.zeros(len(parent1)-crossover_point)])
        
        # Apply symmetry parameter
        if random.random() < alpha:
            # Symmetric recombination
            child = np.where(mask, parent1, parent2)
        else:
            # Asymmetric recombination
            dominance = random.uniform(0.3, 0.7)
            child = dominance * parent1 + (1-dominance) * parent2
        
        # Apply stochasticity with possible quantum effects
        noise = beta * (np.random.random(size=parent1.shape) - 0.5)
        if self.quantum_entanglement:
            noise *= np.random.normal(0, 1, size=parent1.shape)
        
        return child + noise
    
    def apply_mutation(self, genome, mutation_rate):
        """Apply mutations"""
        mask = np.random.random(size=genome.shape) < mutation_rate
        mutations = np.random.normal(0, 0.5, size=genome.shape)
        
        return np.where(mask, genome + mutations, genome)
    
    def inherit_epigenetic_tags(self, parent1_tags, parent2_tags):
        """Enhanced epigenetic inheritance with memory effects"""
        # Basic inheritance - random choice from parents
        child_tags = np.where(np.random.randint(0, 2, size=parent1_tags.shape), parent1_tags, parent2_tags)
        
        # Apply some random epigenetic mutations with memory
        mutation_mask = np.random.random(size=child_tags.shape) < 0.05
        if self.neural_interface:
            # Neural interface adds correlation to mutations
            correlated_mask = mutation_mask | (np.random.random(size=child_tags.shape) < 0.1)
            child_tags = np.where(correlated_mask, 1 - child_tags, child_tags)
        else:
            child_tags = np.where(mutation_mask, 1 - child_tags, child_tags)
        
        return child_tags
    
    def apply_developmental_plasticity(self, population):
        """Apply developmental plasticity with environmental interactions"""
        if not hasattr(self, 'epigenetic_tags') or not self.epigenetic_tags:
            return population
        
        modified_population = []
        for i, genome in enumerate(population):
            if i in self.epigenetic_tags:
                # Environmentally-sensitive epigenetic modifications
                env_factor = 1 + 0.5 * np.sin(self.current_generation / 10)  # Simulated environmental cycle
                modified_genome = genome * (1 - env_factor * self.developmental_plasticity * self.epigenetic_tags[i])
                modified_population.append(modified_genome)
            else:
                modified_population.append(genome)
        
        return np.array(modified_population)
    
    def calculate_hybrid_vigor(self, parents, fitness):
        """Calculate hybrid vigor using theorem-derived formula"""
        if len(parents) < 2:
            return 0.0
            
        # Get parent genomes
        parent1 = self.population[parents[0]]
        parent2 = self.population[parents[1]]
        
        # Calculate distance between parents
        distance = np.linalg.norm(parent1 - parent2)
        
        # Calculate variances
        var1 = np.var(parent1)
        var2 = np.var(parent2)
        
        # Hybrid vigor formula: ΔF = κ·e^{-λd(g₁,g₂)}·min(Var(g₁),Var(g₂))
        kappa = 0.5  # Scaling constant
        lamda = 0.1  # Distance decay rate
        hybrid_vigor = kappa * np.exp(-lamda * distance) * min(var1, var2)
        
        # Add epigenetic enhancement if available
        if hasattr(self, 'epigenetic_tags'):
            epigenetic_diversity = self.calculate_epigenetic_variation()
            hybrid_vigor *= (1 + 0.3 * epigenetic_diversity)  # Reduced from 0.5 to 0.3 to avoid overemphasis
        
        return max(0, hybrid_vigor)  # Ensure non-negative
    
    def check_kill_switch_conditions(self, population, fitness):
        """Enhanced kill-switch conditions with multiple metrics"""
        # Check lineage depth - only if tracking is enabled
        if self.params.get("lineage_tracking", True):
            lineage_depth = self.calculate_lineage_depth()
            if lineage_depth > self.params.get("lineage_depth_limit", 200):  # Increased from 50
                return True
        
        # Check fitness collapse - handle both scalar and array fitness
        if np.isscalar(fitness):
            if not np.isnan(fitness) and fitness < self.params.get("fitness_collapse_threshold", -5000):
                return True
        else:  # Array-like case
            if len(fitness) > 0:
                mean_fitness = np.mean(fitness)
                if not np.isnan(mean_fitness) and mean_fitness < self.params.get("fitness_collapse_threshold", -5000):
                    return True
        
        # Check diversity collapse - only if population has diversity
        if len(population) > 1:
            diversity = self.calculate_diversity(population)
            if not np.isnan(diversity) and diversity < self.params.get("diversity_threshold", 0.01):  # Lower threshold
                return True
        
        # Check quantum decoherence if enabled
        if self.quantum_entanglement:
            quantum_coherence = self.calculate_quantum_entanglement()
            if not np.isnan(quantum_coherence) and quantum_coherence < 0.01:  # More strict decoherence threshold
                return True
        
        return False
    
    def pause(self):
        self.paused = True
        
    def resume(self):
        self.paused = False
        
    def stop(self):
        self.stopped = True


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#2c3e50')
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        # Set plot aesthetics
        self.axes.set_facecolor('#2c3e50')
        self.axes.tick_params(axis='x', colors='#ecf0f1', labelsize=8)
        self.axes.tick_params(axis='y', colors='#ecf0f1', labelsize=8)
        for spine in self.axes.spines.values():
            spine.set_color('#ecf0f1')
        self.axes.xaxis.label.set_color('#ecf0f1')
        self.axes.yaxis.label.set_color('#ecf0f1')
        self.axes.title.set_color('#ecf0f1')


class PopulationVisualization(MplCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent, width, height, dpi)
        self.population_data = None
        self.pca = PCA(n_components=2)
        
    def update_population(self, population):
        """Update the population visualization"""
        self.population_data = population
        self.draw_population()
        
    def draw_population(self):
        """Draw the current population state"""
        if self.population_data is None or len(self.population_data) < 2:
            return
            
        self.axes.clear()
        
        # Reduce dimensionality for visualization
        if self.population_data.shape[1] > 2:
            reduced = self.pca.fit_transform(self.population_data)
        else:
            reduced = self.population_data
        
        # Plot population
        self.axes.scatter(reduced[:,0], reduced[:,1], c='#3498db', alpha=0.6, edgecolors='w', linewidths=0.5)
        
        # Add convex hull if enough points
        if len(reduced) > 2:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(reduced)
                for simplex in hull.simplices:
                    self.axes.plot(reduced[simplex, 0], reduced[simplex, 1], 'r--', alpha=0.3)
            except:
                pass
        
        self.axes.set_title('Population Distribution', color='#ecf0f1', fontsize=10)
        self.axes.set_xlabel('PC1' if self.population_data.shape[1] > 2 else 'Gene 1', color='#ecf0f1', fontsize=8)
        self.axes.set_ylabel('PC2' if self.population_data.shape[1] > 2 else 'Gene 2', color='#ecf0f1', fontsize=8)
        self.axes.grid(True, color='#4a6572', linestyle=':', alpha=0.5)
        
        # Adjust layout to prevent label overlap
        self.figure.tight_layout()
        self.draw()


class ClassicalMatingKernelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Classical Mating Kernel System v1.0")
        self.setStyleSheet("background-color: #2c3e50; color: #ecf0f1;")
        
        # Window state management
        self.setWindowFlags(Qt.Window | 
                          Qt.WindowMaximizeButtonHint |
                          Qt.WindowMinimizeButtonHint |
                          Qt.WindowCloseButtonHint)
        self.showMaximized()
        self.setWindowState(Qt.WindowMaximized)
        
        # Create menu bar
        self.create_menu()
        
        # Initialize simulation variables
        self.simulation_thread = None
        self.generation_data = []
        self.lineage_data = {}
        self.epigenetic_data = {}
        self.population_history = []
        
        # Create main tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #3498db; 
                background: #34495e; 
            }
            QTabBar::tab {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 10px;
                border: 1px solid #3498db;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
        """)
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.simulation_tab = QWidget()
        self.visualization_tab = QWidget()
        self.population_tab = QWidget()
        self.lineage_tab = QWidget()
        self.epigenetics_tab = QWidget()
        self.documentation_tab = QWidget()
        
        self.tabs.addTab(self.simulation_tab, "Simulation")
        self.tabs.addTab(self.visualization_tab, "Metrics")
        self.tabs.addTab(self.population_tab, "Population")
        self.tabs.addTab(self.lineage_tab, "Lineage")
        self.tabs.addTab(self.epigenetics_tab, "Epigenetics")
        self.tabs.addTab(self.documentation_tab, "Documentation")
        
        self.setup_simulation_tab()
        self.setup_visualization_tab()
        self.setup_population_tab()
        self.setup_lineage_tab()
        self.setup_epigenetics_tab()
        self.setup_documentation_tab()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #34495e; color: #ecf0f1;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Initialize plots
        self.initialize_plots()
    
    def setup_simulation_tab(self):
        layout = QVBoxLayout(self.simulation_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Create parameter group
        param_group = QGroupBox("Kernel Parameters")
        param_group.setStyleSheet("""
            QGroupBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 1ex;
                font-weight: bold;
                color: #ecf0f1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        param_layout = QVBoxLayout(param_group)
        
        # Parameter grid
        param_grid = QWidget()
        grid_layout = QHBoxLayout(param_grid)
        
        # Left column
        left_col = QVBoxLayout()
        left_col.setSpacing(15)
        self.alpha_slider = self.create_parameter_slider("α (Symmetry)", 0.5, 0.0, 1.0)
        self.beta_slider = self.create_parameter_slider("β (Stochasticity)", 1.0, 0.0, 5.0)
        self.gamma_slider = self.create_parameter_slider("γ (Granularity)", 0.3, 0.0, 1.0)
        left_col.addWidget(self.alpha_slider)
        left_col.addWidget(self.beta_slider)
        left_col.addWidget(self.gamma_slider)
        
        # Right column
        right_col = QVBoxLayout()
        right_col.setSpacing(15)
        self.delta_slider = self.create_parameter_slider("δ (Compatibility)", 0.7, 0.0, 1.0)
        self.epsilon_slider = self.create_parameter_slider("ε (Fluke Contingency)", 0.05, 0.0, 0.5)
        self.zeta_slider = self.create_parameter_slider("ζ (Speciation Temp)", 1.0, 0.0, 5.0)
        right_col.addWidget(self.delta_slider)
        right_col.addWidget(self.epsilon_slider)
        right_col.addWidget(self.zeta_slider)
        
        # Add columns to grid
        grid_layout.addLayout(left_col)
        grid_layout.addLayout(right_col)
        param_layout.addWidget(param_grid)
        
        # Advanced parameters group
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_group.setStyleSheet(param_group.styleSheet())
        advanced_layout = QHBoxLayout(advanced_group)
        advanced_layout.setSpacing(15)
        
        # Advanced parameters
        self.lineage_check = QCheckBox("Enable Lineage Tracking")
        self.lineage_check.setChecked(True)
        self.lineage_check.setStyleSheet("color: #ecf0f1;")
        
        self.epigenetic_check = QCheckBox("Enable Epigenetic System")
        self.epigenetic_check.setChecked(True)
        self.epigenetic_check.setStyleSheet("color: #ecf0f1;")
        
        self.neural_check = QCheckBox("Neural Interface")
        self.neural_check.setChecked(False)
        self.neural_check.setStyleSheet("color: #ecf0f1;")
        
        self.topology_check = QCheckBox("Topological Features")
        self.topology_check.setChecked(False)
        self.topology_check.setStyleSheet("color: #ecf0f1;")
        
        self.plasticity_spin = self.create_double_spinbox("Developmental Plasticity", 0.1, 0.0, 1.0)
        self.depth_limit_spin = self.create_spinbox("Lineage Depth Limit", 50, 1, 1000)
        
        advanced_layout.addWidget(self.lineage_check)
        advanced_layout.addWidget(self.epigenetic_check)
        advanced_layout.addWidget(self.neural_check)
        advanced_layout.addWidget(self.topology_check)
        advanced_layout.addWidget(self.plasticity_spin)
        advanced_layout.addWidget(self.depth_limit_spin)
        
        param_layout.addWidget(advanced_group)
        
        # Population parameters
        pop_group = QGroupBox("Population Parameters")
        pop_group.setStyleSheet(param_group.styleSheet())
        pop_layout = QHBoxLayout(pop_group)
        pop_layout.setSpacing(15)
        
        # UI Spinboxes for new parameters
        self.pop_floor_spin = self.create_spinbox("Population Floor", 20, 5, 1000)
        self.archive_rate_spin = self.create_double_spinbox("Archive Rate %", 15, 0, 50)
        self.epi_pulse_spin = self.create_spinbox("Epigenetic Pulse", 10, 1, 100)
        self.sigma_ref_spin = self.create_double_spinbox("σ_ref (Variance Ref)", 1.0, 0.1, 10.0)
        self.tau_epsilon_spin = self.create_double_spinbox("τ_ε (Gate Sensitivity)", 2.0, 0.1, 10.0)
        
        # Existing population parameters
        self.pop_size_spin = self.create_spinbox("Population Size", 100, 10, 1000)
        self.generations_spin = self.create_spinbox("Generations", 50, 1, 500)
        self.genome_dim_spin = self.create_spinbox("Genome Dimension", 20, 1, 100)
        self.mutation_spin = self.create_double_spinbox("Mutation Rate", 0.01, 0.0, 0.2)
        self.speed_spin = self.create_double_spinbox("Speed", 0.1, 0.01, 2.0)
        
        pop_layout.addWidget(self.pop_size_spin)
        pop_layout.addWidget(self.generations_spin)
        pop_layout.addWidget(self.genome_dim_spin)
        pop_layout.addWidget(self.mutation_spin)
        pop_layout.addWidget(self.speed_spin)
        pop_layout.addWidget(self.pop_floor_spin)
        pop_layout.addWidget(self.archive_rate_spin)
        pop_layout.addWidget(self.epi_pulse_spin)
        pop_layout.addWidget(self.sigma_ref_spin)
        pop_layout.addWidget(self.tau_epsilon_spin)
        
        param_layout.addWidget(pop_group)
        layout.addWidget(param_group)
        
        # Simulation controls
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.start_btn.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet(self.get_button_style("#f39c12"))
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(self.get_button_style("#e74c3c"))
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setStyleSheet(self.get_button_style("#2ecc71"))
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)
        
        layout.addLayout(control_layout)
        
        # Metrics table
        table_group = QGroupBox("Simulation Metrics")
        table_group.setStyleSheet(param_group.styleSheet())
        table_layout = QVBoxLayout(table_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setStyleSheet("""
            QTableWidget {
                background-color: #34495e;
                color: #ecf0f1;
                gridline-color: #3498db;
                border: none;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: none;
            }
        """)
        self.metrics_table.setColumnCount(13)
        self.metrics_table.setHorizontalHeaderLabels([
            "Gen", "Avg Fit", "Best Fit", 
            "Div", "Topo Div", "Hybrid Vig", 
            "Fluke", "Entropy", "Lineage", 
            "Epigen", "Develop", "ζ_t", "λ_t"
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        
        table_layout.addWidget(self.metrics_table)
        layout.addWidget(table_group)
    
    def setup_visualization_tab(self):
        layout = QVBoxLayout(self.visualization_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create properly sized matplotlib canvas with navigation toolbar
        self.canvas = MplCanvas(self, width=7, height=5, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create a frame for the plot to control size
        plot_frame = QFrame()
        plot_frame.setFrameShape(QFrame.StyledPanel)
        plot_frame.setStyleSheet("background-color: #34495e;")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        layout.addWidget(plot_frame)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.update_btn = QPushButton("Update Plots")
        self.update_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.update_btn.clicked.connect(self.update_plots)
        control_layout.addWidget(self.update_btn)
        
        self.export_btn = QPushButton("Export Plots")
        self.export_btn.setStyleSheet(self.get_button_style("#2ecc71"))
        self.export_btn.clicked.connect(self.export_plots)
        control_layout.addWidget(self.export_btn)
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Fitness Metrics", "Diversity Metrics", "All Metrics", "Adaptive Parameters"])
        self.plot_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        self.plot_type_combo.currentIndexChanged.connect(self.update_plots)
        control_layout.addWidget(self.plot_type_combo)
        
        layout.addLayout(control_layout)
    
    def setup_population_tab(self):
        layout = QVBoxLayout(self.population_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create population visualization canvas
        self.pop_canvas = PopulationVisualization(self, width=7, height=5, dpi=100)
        self.pop_toolbar = NavigationToolbar(self.pop_canvas, self)
        
        # Create a frame for the population plot
        pop_frame = QFrame()
        pop_frame.setFrameShape(QFrame.StyledPanel)
        pop_frame.setStyleSheet("background-color: #34495e;")
        pop_layout = QVBoxLayout(pop_frame)
        pop_layout.setContentsMargins(0, 0, 0, 0)
        pop_layout.addWidget(self.pop_toolbar)
        pop_layout.addWidget(self.pop_canvas)
        
        layout.addWidget(pop_frame)
        
        # Generation slider
        self.gen_slider = QSlider(Qt.Horizontal)
        self.gen_slider.setRange(0, 0)
        self.gen_slider.valueChanged.connect(self.show_population_snapshot)
        self.gen_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #34495e;
                border: 1px solid #3498db;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border: 1px solid #ecf0f1;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        
        self.gen_label = QLabel("Generation: 0")
        self.gen_label.setStyleSheet("color: #ecf0f1;")
        
        layout.addWidget(self.gen_label)
        layout.addWidget(self.gen_slider)
    
    def setup_lineage_tab(self):
        layout = QVBoxLayout(self.lineage_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Lineage visualization group
        lineage_group = QGroupBox("Lineage Tracking")
        lineage_group.setStyleSheet("""
            QGroupBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 1ex;
                font-weight: bold;
                color: #ecf0f1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        lineage_layout = QVBoxLayout(lineage_group)
        
        # Lineage hash display
        self.lineage_hash_label = QLabel("Lineage Hash:")
        self.lineage_hash_label.setStyleSheet("color: #ecf0f1;")
        
        self.lineage_hash_display = QLineEdit()
        self.lineage_hash_display.setReadOnly(True)
        self.lineage_hash_display.setStyleSheet("""
            QLineEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        
        # Lineage depth display
        self.lineage_depth_label = QLabel("Current Lineage Depth: 0")
        self.lineage_depth_label.setStyleSheet("color: #ecf0f1;")
        
        # Lineage tree display
        self.lineage_tree_display = QTableWidget()
        self.lineage_tree_display.setStyleSheet("""
            QTableWidget {
                background-color: #34495e;
                color: #ecf0f1;
                gridline-color: #3498db;
                border: none;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: none;
            }
        """)
        self.lineage_tree_display.setColumnCount(3)
        self.lineage_tree_display.setHorizontalHeaderLabels(["Individual", "Parent 1", "Parent 2"])
        self.lineage_tree_display.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lineage_tree_display.verticalHeader().setVisible(False)
        
        lineage_layout.addWidget(self.lineage_hash_label)
        lineage_layout.addWidget(self.lineage_hash_display)
        lineage_layout.addWidget(self.lineage_depth_label)
        lineage_layout.addWidget(self.lineage_tree_display)
        
        layout.addWidget(lineage_group)
        
        # Lineage visualization controls
        control_layout = QHBoxLayout()
        
        self.visualize_lineage_btn = QPushButton("Visualize Lineage Tree")
        self.visualize_lineage_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.visualize_lineage_btn.clicked.connect(self.visualize_lineage_tree)
        control_layout.addWidget(self.visualize_lineage_btn)
        
        self.export_lineage_btn = QPushButton("Export Lineage Data")
        self.export_lineage_btn.setStyleSheet(self.get_button_style("#2ecc71"))
        self.export_lineage_btn.clicked.connect(self.export_lineage_data)
        control_layout.addWidget(self.export_lineage_btn)
        
        layout.addLayout(control_layout)
    
    def setup_epigenetics_tab(self):
        layout = QVBoxLayout(self.epigenetics_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Epigenetics visualization group
        epigenetics_group = QGroupBox("Epigenetic System")
        epigenetics_group.setStyleSheet("""
            QGroupBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 1ex;
                font-weight: bold;
                color: #ecf0f1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        epigenetics_layout = QVBoxLayout(epigenetics_group)
        
        # Epigenetic variation display
        self.epigenetic_var_label = QLabel("Epigenetic Variation: 0.0")
        self.epigenetic_var_label.setStyleSheet("color: #ecf0f1;")
        
        # Epigenetic matrix display
        self.epigenetic_matrix_display = QTableWidget()
        self.epigenetic_matrix_display.setStyleSheet("""
            QTableWidget {
                background-color: #34495e;
                color: #ecf0f1;
                gridline-color: #3498db;
                border: none;
            }
            QHeaderView::section {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: none;
            }
        """)
        self.epigenetic_matrix_display.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.epigenetic_matrix_display.verticalHeader().setVisible(False)
        
        # Epigenetic visualization controls
        control_layout = QHBoxLayout()
        
        self.visualize_epigenetics_btn = QPushButton("Visualize Patterns")
        self.visualize_epigenetics_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.visualize_epigenetics_btn.clicked.connect(self.visualize_epigenetic_patterns)
        control_layout.addWidget(self.visualize_epigenetics_btn)
        
        self.export_epigenetics_btn = QPushButton("Export Data")
        self.export_epigenetics_btn.setStyleSheet(self.get_button_style("#2ecc71"))
        self.export_epigenetics_btn.clicked.connect(self.export_epigenetic_data)
        control_layout.addWidget(self.export_epigenetics_btn)
        
        epigenetics_layout.addWidget(self.epigenetic_var_label)
        epigenetics_layout.addWidget(self.epigenetic_matrix_display)
        epigenetics_layout.addLayout(control_layout)
        
        layout.addWidget(epigenetics_group)

    def setup_documentation_tab(self):
        """Setup the documentation tab with comprehensive theoretical content"""
        layout = QVBoxLayout(self.documentation_tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Create tab widget for documentation sections
        doc_tabs = QTabWidget()
        doc_tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #3498db; 
                background: #34495e; 
            }
            QTabBar::tab {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 8px;
                border: 1px solid #3498db;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
        """)
        
        # Theory tab
        theory_tab = QWidget()
        theory_layout = QVBoxLayout(theory_tab)
        
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_text.setStyleSheet("""
            QTextEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 5px;
            }
        """)
        
        # Load markdown content
        theory_md = """
# Computational Theory of AI Copulation and Hybridization

## Foundational Definitions

Each agent is formally defined by the tuple:
A = 〈g, ι, τ, ρ〉 where:
- g: Genotype (finite encoding of weights/circuits/programs)
- ι: Immutable lineage hash (cryptographic ancestry trace)  
- τ: Epigenetic tag (reversible phenotypic switch)
- ρ: Fitness scalar (environmentally determined)

## Mating Kernels

### Classical Kernel (κ_class)
κ_class: G×G×Θ(6) → ∆(G)  
Parameters:
1. α ∈ [0,1] - Symmetry (parental contribution balance)
2. β ∈ ℝ⁺ - Stochasticity (mutation intensity)
3. γ ∈ {gene,block,module} - Granularity (recombination resolution)  
4. δ > 0 - Compatibility radius
5. ε ∈ [0,1] - Fluke contingency (novelty injection)
6. ζ ∈ ℝ⁺ - Speciation temperature (selection pressure)

### Quantum Kernel (κ_quant)
κ_quant: H_G×H_G×Θ(7) → D(H_G)  
Adds compressibility parameter η ∈ [0,1] for:
- η > 0.7: Semantic fusion
- η < 0.3: Syntactic recombination

## Foundational Theorems

1. **No-Free-Mating Theorem**  
   ∀κ_univ, complexity(κ_univ) → ∞  
   (Universal copulation requires unbounded complexity)

2. **Hybrid Vigor Bound**  
   E[ΔF] ≤ κ·C(g₁,g₂)·min(Var(g₁),Var(g₂))  
   (Fitness gain bounded by parental variance)

3. **Lineage Entropy Growth**  
   H(L_t) = Θ(log t)  
   (Guaranteed novelty emergence)

## Implementation Framework

### Core Components:
1. **Developmental Plasticity**  
   φ(g,E,f) = δ(g,E,f;τ)  
   (Environmentally-modulated phenotype expression)

2. **Symbol-Integrity Governance**  
   Formal registry ensuring consistent interpretation  
   across classical/quantum regimes

3. **Kill-Switch Protocol**  
   Terminates lineages when:  
   - Fitness collapse detected  
   - Diversity threshold violated  
   - Quantum decoherence occurs  
   - Lineage depth exceeded

### Horizontal Transfer Mechanisms:
1. Lateral Gene Transfer (L)  
   L: G×G → ∆(G)  
2. Co-evolutionary Coupling (C)  
   C: P×P → ∆(P)

## Operational Modes

| Mode          | α   | β   | γ     | δ   | ε   | ζ   | η   |
|---------------|-----|-----|-------|-----|-----|-----|-----|
| UniformCross  | 0.5 | 1.0 | gene  | ∞   | 0.0 | ∞   | -   |
| Asymmetric    | 0.0 | 0.2 | module| δ₀  | 0.0 | ζ₀  | -   |  
| QuantumFusion | 0.5 | 1.0 | gene  | ∞   | 0.0 | ∞   | 0.8 |
"""
        theory_html = markdown2.markdown(theory_md)
        theory_text.setHtml(theory_html)
        
        theory_layout.addWidget(theory_text)
        doc_tabs.addTab(theory_tab, "Theory")
        
        # Implementation tab
        impl_tab = QWidget()
        impl_layout = QVBoxLayout(impl_tab)
        
        impl_text = QTextEdit()
        impl_text.setReadOnly(True)
        impl_text.setStyleSheet(theory_text.styleSheet())
        
        impl_md = """
# Formal Implementation

## Mathematical Foundations

### Genotype Space (S)
S = (G, Σ_G, d) where:
- G: Set of admissible genotypes
- Σ_G: σ-algebra over G  
- d: Compatibility metric:
  - ‖g₁ - g₂‖_p for vectorial
  - edit distance for structural

### Kernel Axioms
1. **Symmetry**: κ(g₁,g₂,θ) = κ(g₂,g₁,θ) when α=0.5
2. **Fluke-Admissibility**: supp(κ) ⊆ B_ε(G_viable)  
3. **Lineage Consistency**: E[d(g′,g₁)] ≤ λd(g₁,g₂) + ε

## Quantum Extensions

### Hilbert Space Embedding
H_G = {|g〉 | g∈G} with 〈g_i|g_j〉 = exp(-d(g_i,g_j)²/2σ²)

### Entanglement Operator
U_η = exp(-iηH_ent)  
H_ent = Σ_k φ_k|ψ_k〉〈ψ_k|  
where φ_k = f(MI(g₁^(k),g₂^(k))) (quantum mutual information)

## Population Dynamics

### Evolution Equation
P_{t+1}(A) = (1-μ)∫∫κ(A|g₁,g₂,θ)dS_t(g₁,g₂) + μM(A)

where:
- S_t: Mate-selection distribution ∝ e^{ζ⁻¹d(g₁,g₂)}f(g₁)f(g₂)
- μ: Mutation rate
- M: Mutation kernel

### Theorem Compliance
1. **Hybrid Vigor Bound**:  
   E[ΔF] ≤ κ·C(g₁,g₂)·min(σ²_f(g₁),σ²_f(g₂))  
   C(g₁,g₂) = e^{-λd(g₁,g₂)}

2. **Viability Threshold**:  
   P_viable ≥ 1 - exp(-ξ/(δ-δ_min)) ∀δ > δ_min

## Decoherence Resilience
Implementation requires:
̂κ_quant = P_code ◦ U_η ◦ E_err  
satisfying:
F(̂κ_quant(ρ),̂κ_quant(σ)) ≥ 1 - e^{-ξd(ρ,σ)}  
for all density operators ρ,σ ∈ D(H_G)
"""
        impl_html = markdown2.markdown(impl_md)
        impl_text.setHtml(impl_html)
        
        impl_layout.addWidget(impl_text)
        doc_tabs.addTab(impl_tab, "Implementation")
        
        # API tab
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        
        api_text = QTextEdit()
        api_text.setReadOnly(True)
        api_text.setStyleSheet(theory_text.styleSheet())
        
        api_md = """
# API Reference

## Core Classes

### SimulationThread

The main simulation worker class with key methods:

- `run()`: Main simulation loop
- `mating_kernel()`: Implements the core recombination logic
- `evaluate_fitness()`: Calculates agent fitness
- `select_parents()`: Performs selection operation

Key signals:
- `update_signal`: Emits generation data
- `finished_signal`: Emits when simulation completes
- `kill_switch_activated`: Emits when kill conditions met

### ClassicalMatingKernelApp

The main application class with key methods:

- `start_simulation()`: Initializes and starts simulation
- `update_plots()`: Updates visualization
- `save_results()`: Exports simulation data

## Data Structures

### Generation Data

Each generation produces a dictionary with:
- Generation number
- Average and best fitness
- Diversity metrics
- Lineage depth
- Epigenetic variation
- Other advanced metrics

### Lineage Tracking

The lineage system maintains:
- Cryptographic hashes for each individual
- Parent-child relationships
- Depth calculations

## Extending the System

To add new features:

1. **New Parameters**:
   - Add UI controls in setup_simulation_tab()
   - Include in params dictionary

2. **New Algorithms**:
   - Extend SimulationThread methods
   - Add new calculation methods

3. **New Visualizations**:
   - Create new canvas classes
   - Add to appropriate tab
"""
        api_html = markdown2.markdown(api_md)
        api_text.setHtml(api_html)
        
        api_layout.addWidget(api_text)
        doc_tabs.addTab(api_tab, "API")
        
        # Add documentation tabs to layout
        layout.addWidget(doc_tabs)
    
    def create_menu(self):
        """Create the main menu system"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #3498db;
            }
            QMenu {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
            }
            QMenu::item:selected {
                background-color: #3498db;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = file_menu.addAction("New Simulation")
        new_action.triggered.connect(self.new_simulation)
        
        open_action = file_menu.addAction("Open...")
        open_action.triggered.connect(self.open_simulation)
        
        save_action = file_menu.addAction("Save Results")
        save_action.triggered.connect(self.save_results)
        
        export_menu = file_menu.addMenu("Export")
        export_lineage_action = export_menu.addAction("Lineage Data")
        export_lineage_action.triggered.connect(self.export_lineage_data)
        export_epigenetic_action = export_menu.addAction("Epigenetic Data")
        export_epigenetic_action.triggered.connect(self.export_epigenetic_data)
        export_population_action = export_menu.addAction("Population Snapshots")
        export_population_action.triggered.connect(self.export_population_data)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Simulation menu
        sim_menu = menubar.addMenu("Simulation")
        
        start_action = sim_menu.addAction("Start")
        start_action.triggered.connect(self.start_simulation)
        
        pause_action = sim_menu.addAction("Pause/Resume")
        pause_action.triggered.connect(self.toggle_pause)
        
        stop_action = sim_menu.addAction("Stop")
        stop_action.triggered.connect(self.stop_simulation)
        
        # Visualization menu
        vis_menu = menubar.addMenu("Visualization")
        
        update_plots_action = vis_menu.addAction("Update Plots")
        update_plots_action.triggered.connect(self.update_plots)
        
        export_plots_action = vis_menu.addAction("Export Plots")
        export_plots_action.triggered.connect(self.export_plots)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        
        lineage_analysis_action = analysis_menu.addAction("Lineage Analysis")
        lineage_analysis_action.triggered.connect(self.show_lineage_analysis)
        
        epigenetic_analysis_action = analysis_menu.addAction("Epigenetic Analysis")
        epigenetic_analysis_action.triggered.connect(self.show_epigenetic_analysis)
        
        population_analysis_action = analysis_menu.addAction("Population Analysis")
        population_analysis_action.triggered.connect(self.show_population_analysis)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)
        
        docs_action = help_menu.addAction("Documentation")
        docs_action.triggered.connect(self.show_documentation)
        
        examples_action = help_menu.addAction("Examples")
        examples_action.triggered.connect(self.show_examples)
        
        help_menu.addSeparator()
        
        website_action = help_menu.addAction("Repository")
        website_action.triggered.connect(lambda: webbrowser.open("https://github.com/Artwell-XE/ai-copulation-theory"))
    
    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #7f8c8d;
            }}
        """
    
    def lighten_color(self, hex_color, factor=0.3):
        """Lighten a color by a given factor (0-1)"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = min(255, r + int((255 - r) * factor))
        g = min(255, g + int((255 - g) * factor))
        b = min(255, b + int((255 - b) * factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def create_parameter_slider(self, label, default, min_val, max_val):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(label)
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)
        
        slider_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * 100))
        slider.setMaximum(int(max_val * 100))
        slider.setValue(int(default * 100))
        slider_layout.addWidget(slider)
        
        value_label = QLabel(f"{default:.2f}")
        value_label.setStyleSheet("min-width: 40px; text-align: center;")
        value_label.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(value_label)
        
        # Connect slider to update label
        slider.valueChanged.connect(lambda val: value_label.setText(f"{val/100:.2f}"))
        
        layout.addLayout(slider_layout)
        return widget
    
    def create_spinbox(self, label, default, min_val, max_val):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(label)
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)
        
        # Use QSpinBox for integer values
        spinbox = QSpinBox()
        spinbox.setValue(default)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        layout.addWidget(spinbox)
        
        return widget
    
    def create_double_spinbox(self, label, default, min_val, max_val):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel(label)
        lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl)
        
        spinbox = QDoubleSpinBox()
        spinbox.setValue(default)
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setSingleStep(0.01)
        spinbox.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        layout.addWidget(spinbox)
        
        return widget
    
    def initialize_plots(self):
        """Initialize plots with empty data and proper sizing"""
        self.canvas.axes.clear()
        
        # Set plot properties with better spacing
        self.canvas.axes.set_title('Simulation Metrics', color='#ecf0f1', fontsize=10)
        self.canvas.axes.set_xlabel('Generation', color='#ecf0f1', fontsize=8)
        self.canvas.axes.set_ylabel('Value', color='#ecf0f1', fontsize=8)
        self.canvas.axes.tick_params(axis='both', which='major', labelsize=8)
        self.canvas.axes.grid(True, color='#4a6572', linestyle=':', alpha=0.5)
        
        # Adjust layout to prevent overlap
        self.canvas.figure.tight_layout()
        self.canvas.draw()
    
    def show_about(self):
        """Show the About dialog with enhanced content"""
        about_text = """
        <h2>Classical Mating Kernel System v1.0</h2>
        <p><b>Implementation of the Computational Theory of AI Copulation and Hybridization</b></p>
        
        <h3>Core Features</h3>
        <ul>
            <li>Advanced mating kernel with 6-dimensional parameter space</li>            
            <li>Neural interface for social-genetic interactions</li>
            <li>Topological diversity metrics</li>
            <li>Comprehensive lineage tracking with cryptographic hashing</li>
            <li>Epigenetic system with developmental plasticity</li>
        </ul>
        
        <h3>Theoretical Foundations</h3>
        <p>The system implements the formal theory described in:</p>
        <p><i>"On the Computational Theory of AI Copulation & Hybridization"</i><br>
        Liberty A. Mareya, 2025</p>
        
        <h3>System Requirements</h3>
        <ul>
            <li>Python 3.8+</li>
            <li>PyQt5</li>
            <li>NumPy, SciPy, pandas</li>
            <li>Matplotlib</li>
            <li>NetworkX (for lineage visualization)</li>
        </ul>
        
        <h3>License</h3>
        <p>GNU General Public License v3.0</p>
        
        <p>© 2025 Computational Theory of AI Copulation Research Group</p>
        
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("About Classical Mating Kernel")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()

    def show_documentation(self):
        """Switch to documentation tab"""
        self.tabs.setCurrentIndex(5)  # Documentation tab is index 5
        self.status_bar.showMessage("Showing documentation")
    
    def show_examples(self):
        """Show examples dialog"""
        examples_text = """
        <h2>Example Configurations</h2>
        
        <h3>Basic Evolutionary Optimization</h3>
        <pre>
        α = 0.5 (Balanced symmetry)
        β = 1.0 (Moderate stochasticity)
        γ = 0.3 (Gene-level recombination)
        δ = 0.7 (Standard compatibility)
        ε = 0.05 (Low fluke rate)
        ζ = 1.0 (Neutral speciation)
        </pre>
        
        <h3>Speciation Experiment</h3>
        <pre>
        α = 0.7 (Asymmetric recombination)
        β = 0.5 (Low stochasticity)
        γ = 0.8 (Module-level recombination)
        δ = 0.9 (High compatibility)
        ε = 0.01 (Very low fluke rate)
        ζ = 0.3 (Strong speciation pressure)
        </pre>
        
        <h3>Innovation Search</h3>
        <pre>
        α = 0.3 (Highly asymmetric)
        β = 2.0 (High stochasticity)
        γ = 0.2 (Fine-grained recombination)
        δ = 0.5 (Medium compatibility)
        ε = 0.1 (Elevated fluke rate)
        ζ = 2.0 (Weak selection pressure)
        </pre>
        
    
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("Example Configurations")
        msg.setTextFormat(Qt.RichText)
        msg.setText(examples_text)
        msg.exec_()
    
    def show_lineage_analysis(self):
        """Switch to lineage tab"""
        self.tabs.setCurrentIndex(3)
        self.status_bar.showMessage("Showing lineage analysis")
    
    def show_epigenetic_analysis(self):
        """Switch to epigenetics tab"""
        self.tabs.setCurrentIndex(4)
        self.status_bar.showMessage("Showing epigenetic analysis")
    
    def show_population_analysis(self):
        """Switch to population tab"""
        self.tabs.setCurrentIndex(2)
        self.status_bar.showMessage("Showing population analysis")
    
    def new_simulation(self):
        """Reset simulation parameters"""
        self.stop_simulation()
        self.generation_data = []
        self.lineage_data = {}
        self.epigenetic_data = {}
        self.population_history = []
        self.metrics_table.setRowCount(0)
        self.lineage_tree_display.setRowCount(0)
        self.epigenetic_matrix_display.setRowCount(0)
        self.gen_slider.setRange(0, 0)
        self.gen_label.setText("Generation: 0")
        self.initialize_plots()
        self.pop_canvas.update_population(None)
        self.status_bar.showMessage("New simulation ready")
    
    def open_simulation(self):
        """Load simulation from file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Simulation", "", "JSON Files (*.json);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.generation_data = data.get("generation_data", [])
                self.lineage_data = data.get("lineage_data", {})
                self.epigenetic_data = data.get("epigenetic_data", {})
                self.population_history = data.get("population_history", [])
                
                # Update metrics table
                self.metrics_table.setRowCount(0)
                for gen_data in self.generation_data:
                    row = self.metrics_table.rowCount()
                    self.metrics_table.insertRow(row)
                    
                    self.metrics_table.setItem(row, 0, QTableWidgetItem(str(gen_data["generation"] + 1)))
                    self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{gen_data['avg_fitness']:.4f}"))
                    self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{gen_data['best_fitness']:.4f}"))
                    self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{gen_data['diversity']:.4f}"))
                    self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{gen_data.get('topological_diversity', 0):.4f}"))
                    self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{gen_data['hybrid_vigor']:.4f}"))
                    self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{gen_data['fluke_rate']:.4f}"))
                    self.metrics_table.setItem(row, 7, QTableWidgetItem(f"{gen_data['entropy']:.4f}"))
                    self.metrics_table.setItem(row, 8, QTableWidgetItem(f"{gen_data.get('lineage_depth', 0)}"))
                    self.metrics_table.setItem(row, 9, QTableWidgetItem(f"{gen_data.get('epigenetic_variation', 0):.4f}"))
                    self.metrics_table.setItem(row, 10, QTableWidgetItem(f"{gen_data.get('developmental_effect', 0):.4f}"))
                    self.metrics_table.setItem(row, 11, QTableWidgetItem(f"{gen_data.get('quantum_entanglement', 0):.4f}"))
                    self.metrics_table.setItem(row, 12, QTableWidgetItem(f"{gen_data.get('adaptive_zeta', 0):.4f}"))
                    self.metrics_table.setItem(row, 13, QTableWidgetItem(f"{gen_data.get('adaptive_lambda', 0):.4f}"))
                
                # Update population slider
                self.gen_slider.setRange(0, len(self.population_history)-1)
                if self.population_history:
                    self.gen_slider.setValue(len(self.population_history)-1)
                    self.show_population_snapshot(len(self.population_history)-1)
                
                self.update_plots()
                self.status_bar.showMessage(f"Loaded simulation from {filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load simulation: {str(e)}")
                self.status_bar.showMessage("Failed to load simulation")
    
    def start_simulation(self):
        if self.simulation_thread and self.simulation_thread.isRunning():
            self.status_bar.showMessage("Simulation already running!")
            return
        
        # Get parameters
        params = {
            "α": self.alpha_slider.findChild(QSlider).value() / 100.0,
            "β": self.beta_slider.findChild(QSlider).value() / 100.0,
            "γ": self.gamma_slider.findChild(QSlider).value() / 100.0,
            "δ": self.delta_slider.findChild(QSlider).value() / 100.0,
            "ε": self.epsilon_slider.findChild(QSlider).value() / 100.0,
            "ζ": self.zeta_slider.findChild(QSlider).value() / 100.0,
            "pop_size": self.pop_size_spin.findChild(QSpinBox).value(),
            "generations": self.generations_spin.findChild(QSpinBox).value(),
            "genome_dim": self.genome_dim_spin.findChild(QSpinBox).value(),
            "mutation_rate": self.mutation_spin.findChild(QDoubleSpinBox).value(),
            "speed": self.speed_spin.findChild(QDoubleSpinBox).value(),
            "lineage_tracking": self.lineage_check.isChecked(),
            "epigenetic_system": self.epigenetic_check.isChecked(),
            "neural_interface": self.neural_check.isChecked(),
            "topological_features": self.topology_check.isChecked(),
            "developmental_plasticity": self.plasticity_spin.findChild(QDoubleSpinBox).value(),
            "lineage_depth_limit": self.depth_limit_spin.findChild(QSpinBox).value(),
            "population_floor": self.pop_floor_spin.findChild(QSpinBox).value(),
            "archive_rate": self.archive_rate_spin.findChild(QDoubleSpinBox).value() / 100.0,
            "epi_refresh_period": self.epi_pulse_spin.findChild(QSpinBox).value(),
            "sigma_ref": self.sigma_ref_spin.findChild(QDoubleSpinBox).value(),
            "tau_epsilon": self.tau_epsilon_spin.findChild(QDoubleSpinBox).value()
        }
        
        # Reset simulation data
        self.generation_data = []
        self.lineage_data = {}
        self.epigenetic_data = {}
        self.population_history = []
        self.metrics_table.setRowCount(0)
        self.lineage_tree_display.setRowCount(0)
        self.epigenetic_matrix_display.setRowCount(0)
        self.gen_slider.setRange(0, 0)
        self.gen_label.setText("Generation: 0")
        
        # Update UI controls
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.status_bar.showMessage("Simulation starting...")
        
        # Create and start simulation thread
        self.simulation_thread = SimulationThread(params)
        self.simulation_thread.update_signal.connect(self.update_simulation_data)
        self.simulation_thread.finished_signal.connect(self.simulation_finished)
        self.simulation_thread.kill_switch_activated.connect(self.kill_switch_activated)
        self.simulation_thread.population_snapshot.connect(self.update_population_visualization)
        self.simulation_thread.start()
    
    def toggle_pause(self):
        if not self.simulation_thread:
            return
            
        if self.simulation_thread.paused:
            self.simulation_thread.resume()
            self.pause_btn.setText("Pause")
            self.status_bar.showMessage("Simulation resumed")
        else:
            self.simulation_thread.pause()
            self.pause_btn.setText("Resume")
            self.status_bar.showMessage("Simulation paused")
    
    def stop_simulation(self):
        if self.simulation_thread:
            self.simulation_thread.stop()
            self.simulation_thread.wait()
            self.simulation_thread = None
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("Simulation stopped")
    
    def kill_switch_activated(self):
        """Handle kill-switch activation"""
        self.stop_simulation()
        QMessageBox.warning(self, "Kill-Switch Activated", 
                           "The kill-switch protocol has been activated due to:\n"
                           "- Excessive lineage depth\n"
                           "- Fitness collapse\n"
                           "- Diversity collapse\n"
                           "- Quantum decoherence (if enabled)")
        self.status_bar.showMessage("Kill-switch activated - simulation stopped")
    
    def update_simulation_data(self, data):
        """Update UI with new simulation data"""
        self.generation_data.append(data)
        
        # Update metrics table
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        
        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(data["generation"] + 1)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{data['avg_fitness']:.4f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{data['best_fitness']:.4f}"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{data['diversity']:.4f}"))
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{data.get('topological_diversity', 0):.4f}"))
        self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{data['hybrid_vigor']:.4f}"))
        self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{data['fluke_rate']:.4f}"))
        self.metrics_table.setItem(row, 7, QTableWidgetItem(f"{data['entropy']:.4f}"))
        self.metrics_table.setItem(row, 8, QTableWidgetItem(f"{data.get('lineage_depth', 0)}"))
        self.metrics_table.setItem(row, 9, QTableWidgetItem(f"{data.get('epigenetic_variation', 0):.4f}"))
        self.metrics_table.setItem(row, 10, QTableWidgetItem(f"{data.get('developmental_effect', 0):.4f}"))
        self.metrics_table.setItem(row, 11, QTableWidgetItem(f"{data.get('adaptive_zeta', 0):.4f}"))
        self.metrics_table.setItem(row, 12, QTableWidgetItem(f"{data.get('adaptive_lambda', 0):.4f}"))
        
        # Scroll to bottom
        self.metrics_table.scrollToBottom()
        
        # Update lineage tab
        if hasattr(self.simulation_thread, 'lineage_hashes'):
            lineage_hashes = list(self.simulation_thread.lineage_hashes.values())
            if lineage_hashes:
                self.lineage_hash_display.setText(lineage_hashes[0][:16] + "...")
                self.lineage_depth_label.setText(f"Current Lineage Depth: {data.get('lineage_depth', 0)}")
                
                # Update lineage tree display - make thread-safe copy first
                self.lineage_tree_display.setRowCount(0)
                if hasattr(self.simulation_thread, 'lineage_tree'):
                    try:
                        # Create a copy of the lineage tree to avoid concurrent modification
                        lineage_tree_copy = dict(self.simulation_thread.lineage_tree)
                        for child, parents in lineage_tree_copy.items():
                            if parents:  # Ensure there are parents
                                row = self.lineage_tree_display.rowCount()
                                self.lineage_tree_display.insertRow(row)
                                self.lineage_tree_display.setItem(row, 0, QTableWidgetItem(child[:8] + "..."))
                                self.lineage_tree_display.setItem(row, 1, QTableWidgetItem(parents[0][:8] + "..."))
                                self.lineage_tree_display.setItem(row, 2, QTableWidgetItem(parents[1][:8] + "..."))
                    except RuntimeError:
                        # Skip update if dictionary is being modified
                        pass
        
        # Update epigenetics tab
        if hasattr(self.simulation_thread, 'epigenetic_tags'):
            self.epigenetic_var_label.setText(f"Epigenetic Variation: {data.get('epigenetic_variation', 0):.4f}")
            
            # Update epigenetic matrix display
            self.epigenetic_matrix_display.setRowCount(0)
            if self.simulation_thread.epigenetic_tags:
                first_five = dict(list(self.simulation_thread.epigenetic_tags.items())[:5])
                self.epigenetic_matrix_display.setColumnCount(len(next(iter(first_five.values()))))
                self.epigenetic_matrix_display.setHorizontalHeaderLabels([f"Gene {i+1}" for i in range(len(next(iter(first_five.values()))))])
                
                for idx, tags in first_five.items():
                    row = self.epigenetic_matrix_display.rowCount()
                    self.epigenetic_matrix_display.insertRow(row)
                    self.epigenetic_matrix_display.setItem(row, 0, QTableWidgetItem(f"Ind {idx}"))
                    for i, tag in enumerate(tags):
                        self.epigenetic_matrix_display.setItem(row, i+1, QTableWidgetItem("On" if tag else "Off"))
        
        # Update status
        self.status_bar.showMessage(
            f"Generation {data['generation'] + 1} - " 
            f"Avg Fitness: {data['avg_fitness']:.2f}, "
            f"Best Fitness: {data['best_fitness']:.2f}"
        )
        
        # Update plots every 5 generations
        if data["generation"] % 5 == 0:
            self.update_plots()
    
    def update_population_visualization(self, population):
        """Update population visualization with new data"""
        self.pop_canvas.update_population(population)
        
        # Store population snapshot every 10 generations
        if hasattr(self.simulation_thread, 'current_generation') and self.simulation_thread.current_generation % 10 == 0:
            self.population_history.append(population.copy())
            self.gen_slider.setRange(0, len(self.population_history)-1)
            self.gen_slider.setValue(len(self.population_history)-1)
            self.gen_label.setText(f"Generation: {self.simulation_thread.current_generation}")
    
    def show_population_snapshot(self, gen_idx):
        """Show a specific population snapshot"""
        if 0 <= gen_idx < len(self.population_history):
            self.pop_canvas.update_population(self.population_history[gen_idx])
            gen_num = gen_idx * 10  # Since we store every 10 generations
            self.gen_label.setText(f"Generation: {gen_num}")
    
    def simulation_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("Simulation completed")
        self.update_plots()
    
    def update_plots(self):
        """Update visualization plots with proper sizing and layout"""
        if not self.generation_data:
            return
        
        # Prepare data
        gens = [d["generation"] for d in self.generation_data]
        avg_fitness = [d["avg_fitness"] for d in self.generation_data]
        best_fitness = [d["best_fitness"] for d in self.generation_data]
        diversity = [d["diversity"] for d in self.generation_data]
        hybrid_vigor = [d["hybrid_vigor"] for d in self.generation_data]
        entropy = [d["entropy"] for d in self.generation_data]
        lineage_depth = [d.get("lineage_depth", 0) for d in self.generation_data]
        epigenetic_var = [d.get("epigenetic_variation", 0) for d in self.generation_data]
        quantum_ent = [d.get("quantum_entanglement", 0) for d in self.generation_data]
        topo_div = [d.get("topological_diversity", 0) for d in self.generation_data]
        zeta_t = [d.get("adaptive_zeta", 0) for d in self.generation_data]
        lambda_t = [d.get("adaptive_lambda", 0) for d in self.generation_data]
        
        # Clear and update plots based on selected type
        self.canvas.axes.clear()
        
        plot_type = self.plot_type_combo.currentText()
        
        if plot_type == "Fitness Metrics":
            # Plot fitness metrics
            self.canvas.axes.plot(gens, avg_fitness, 'b-', linewidth=1.5, label='Avg Fitness')
            self.canvas.axes.plot(gens, best_fitness, 'r-', linewidth=1.5, label='Best Fitness')
            self.canvas.axes.plot(gens, hybrid_vigor, 'g--', linewidth=1, label='Hybrid Vigor')
            
            self.canvas.axes.set_title('Fitness Metrics', color='#ecf0f1', fontsize=10)
            self.canvas.axes.set_ylabel('Fitness', color='#ecf0f1', fontsize=8)
            
        elif plot_type == "Diversity Metrics":
            # Plot diversity metrics
            self.canvas.axes.plot(gens, diversity, 'c-', linewidth=1.5, label='Genetic Diversity')
            self.canvas.axes.plot(gens, topo_div, 'm-', linewidth=1.5, label='Topological Diversity')
            self.canvas.axes.plot(gens, entropy, 'y-', linewidth=1.5, label='Entropy')
            
            self.canvas.axes.set_title('Diversity Metrics', color='#ecf0f1', fontsize=10)
            self.canvas.axes.set_ylabel('Diversity', color='#ecf0f1', fontsize=8)
            
        elif plot_type == "Adaptive Parameters":
            # Plot adaptive parameters
            self.canvas.axes.plot(gens, zeta_t, 'b-', linewidth=1.5, label='ζ (Speciation Temp)')
            self.canvas.axes.plot(gens, lambda_t, 'r-', linewidth=1.5, label='λ (Archive Rate)')
            self.canvas.axes.plot(gens, [d["fluke_rate"] for d in self.generation_data], 'g--', linewidth=1, label='ε (Fluke Rate)')
            
            self.canvas.axes.set_title('Adaptive Parameters', color='#ecf0f1', fontsize=10)
            self.canvas.axes.set_ylabel('Parameter Value', color='#ecf0f1', fontsize=8)
            
        else:  # All Metrics
            # Plot all metrics with two y-axes
            self.canvas.axes.plot(gens, avg_fitness, 'b-', linewidth=1.5, label='Avg Fitness')
            self.canvas.axes.plot(gens, best_fitness, 'r-', linewidth=1.5, label='Best Fitness')
            self.canvas.axes.plot(gens, diversity, 'g-', linewidth=1.5, label='Diversity')
            self.canvas.axes.plot(gens, hybrid_vigor, 'm-', linewidth=1, label='Hybrid Vigor')
            
            ax2 = self.canvas.axes.twinx()
            ax2.plot(gens, lineage_depth, 'y--', linewidth=1, label='Lineage Depth')
            ax2.plot(gens, epigenetic_var, 'c--', linewidth=1, label='Epigenetic Var')
            ax2.plot(gens, quantum_ent, 'w--', linewidth=1, label='Quantum Ent')
            
            ax2.set_ylabel('Depth/Variation', color='#ecf0f1', fontsize=8)
            ax2.tick_params(axis='y', colors='#ecf0f1', labelsize=8)
            
            # Combine legends from both axes
            lines, labels = self.canvas.axes.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.canvas.axes.legend(lines + lines2, labels + labels2, 
                                  facecolor='#34495e', edgecolor='none', 
                                  labelcolor='#ecf0f1', fontsize=8,
                                  bbox_to_anchor=(1.1, 1), loc='upper left')
            
            self.canvas.axes.set_title('All Simulation Metrics', color='#ecf0f1', fontsize=10)
            self.canvas.axes.set_ylabel('Fitness/Diversity', color='#ecf0f1', fontsize=8)
        
        # Common plot properties
        self.canvas.axes.set_xlabel('Generation', color='#ecf0f1', fontsize=8)
        self.canvas.axes.grid(True, color='#4a6572', linestyle=':', alpha=0.5)
        
        # Set plot colors
        self.canvas.axes.title.set_color('#ecf0f1')
        self.canvas.axes.xaxis.label.set_color('#ecf0f1')
        self.canvas.axes.yaxis.label.set_color('#ecf0f1')
        for spine in self.canvas.axes.spines.values():
            spine.set_color('#ecf0f1')
        self.canvas.axes.tick_params(axis='x', colors='#ecf0f1', labelsize=8)
        self.canvas.axes.tick_params(axis='y', colors='#ecf0f1', labelsize=8)
        
        # Adjust layout to prevent overlap
        self.canvas.figure.tight_layout()
        
        # Redraw canvas
        self.canvas.draw()
    
    def visualize_lineage_tree(self):
        """Visualize the lineage tree using NetworkX with enhanced features"""
        if not hasattr(self.simulation_thread, 'lineage_tree') or not self.simulation_thread.lineage_tree:
            QMessageBox.warning(self, "No Data", "No lineage data available to visualize")
            return
            
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            
            # Create a new window
            lineage_window = QMainWindow(self)
            lineage_window.setWindowTitle("Lineage Tree Visualization")
            lineage_window.resize(1000, 800)
            
            # Create central widget with controls
            central_widget = QWidget()
            lineage_window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Add controls
            control_layout = QHBoxLayout()
            
            depth_label = QLabel("Max Depth:")
            depth_label.setStyleSheet("color: #ecf0f1;")
            control_layout.addWidget(depth_label)
            
            depth_spin = QSpinBox()
            depth_spin.setRange(1, 20)
            depth_spin.setValue(5)
            depth_spin.setStyleSheet("""
                QSpinBox {
                    background-color: #34495e;
                    color: #ecf0f1;
                    border: 1px solid #3498db;
                    border-radius: 4px;
                    padding: 4px;
                }
            """)
            control_layout.addWidget(depth_spin)
            
            update_btn = QPushButton("Update View")
            update_btn.setStyleSheet(self.get_button_style("#3498db"))
            control_layout.addWidget(update_btn)
            
            layout.addLayout(control_layout)
            
            # Create matplotlib figure
            fig = Figure(figsize=(10, 8), dpi=100, facecolor='#2c3e50')
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            def draw_tree(max_depth=5):
                ax.clear()
                
                # Check if we have any lineage data at all
                if not hasattr(self.simulation_thread, 'lineage_tree') or not self.simulation_thread.lineage_tree:
                    ax.text(0.5, 0.5, "No lineage data available", 
                           ha='center', va='center', color='#ecf0f1')
                    canvas.draw()
                    return
                
                # Ensure we have current lineage data
                if not hasattr(self.simulation_thread, 'lineage_hashes'):
                    ax.text(0.5, 0.5, "No current generation data", 
                           ha='center', va='center', color='#ecf0f1')
                    canvas.draw()
                    return
                
                # Build filtered graph
                G = nx.DiGraph()
                nodes_to_show = set()
                
                # Start with current generation
                current_gen = set(self.simulation_thread.lineage_hashes.values())
                nodes_to_show.update(current_gen)
                
                # Add ancestors up to max_depth
                for _ in range(max_depth):
                    parents = set()
                    for node in nodes_to_show:
                        if node in self.simulation_thread.lineage_tree:
                            parents.update(self.simulation_thread.lineage_tree[node])
                    nodes_to_show.update(parents)
                
                # Add edges between shown nodes
                edges_added = 0
                for child in nodes_to_show:
                    if child in self.simulation_thread.lineage_tree:
                        for parent in self.simulation_thread.lineage_tree[child]:
                            if parent in nodes_to_show:
                                G.add_edge(parent, child)
                                edges_added += 1
                
                if edges_added == 0:
                    ax.text(0.5, 0.5, "No parent-child relationships found", 
                           ha='center', va='center', color='#ecf0f1')
                    canvas.draw()
                    return
                
                try:
                    # Draw the graph with better layout
                    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
                except:
                    # Fallback to spring layout if graphviz fails
                    pos = nx.spring_layout(G)
                
                # Color nodes by generation
                node_colors = []
                for node in G.nodes():
                    if node in current_gen:
                        node_colors.append('#3498db')  # Current gen - blue
                    else:
                        node_colors.append('#2ecc71')  # Ancestors - green
                
                nx.draw(G, pos, ax=ax, with_labels=True, 
                       labels={n: n[:8] for n in G.nodes()},  # Show first 8 chars
                       node_size=100, 
                       node_color=node_colors, 
                       edge_color='#7f8c8d', 
                       arrowsize=10, 
                       width=1,
                       font_size=8,
                       font_color='#ecf0f1')
                
                # Set plot aesthetics
                ax.set_facecolor('#2c3e50')
                for spine in ax.spines.values():
                    spine.set_color('#ecf0f1')
                ax.tick_params(axis='both', which='both', length=0)
                
                # Add title with stats
                ax.set_title(f"Lineage Tree (Depth: {max_depth}, Nodes: {len(G.nodes())}, Edges: {len(G.edges())}", 
                            color='#ecf0f1')
                
                canvas.draw()
            
            # Connect update button
            update_btn.clicked.connect(lambda: draw_tree(depth_spin.value()))
            
            # Initial draw
            draw_tree()
            
            # Add to layout
            layout.addWidget(canvas)
            
            # Show window
            lineage_window.show()
            
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize lineage tree: {str(e)}")
    
    def visualize_epigenetic_patterns(self):
        """Visualize epigenetic patterns as a heatmap"""
        if not hasattr(self.simulation_thread, 'epigenetic_tags') or not self.simulation_thread.epigenetic_tags:
            QMessageBox.warning(self, "No Data", "No epigenetic data available to visualize")
            return
            
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create a new window
            epi_window = QMainWindow(self)
            epi_window.setWindowTitle("Epigenetic Patterns Visualization")
            epi_window.resize(800, 600)
            
            # Create central widget
            central_widget = QWidget()
            epi_window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Create matplotlib figure
            fig = Figure(figsize=(8, 6), dpi=100, facecolor='#2c3e50')
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Prepare data matrix
            tags = list(self.simulation_thread.epigenetic_tags.values())
            matrix = np.array(tags[:50])  # Show first 50 individuals
            
            # Create custom colormap
            cmap = LinearSegmentedColormap.from_list('epigenetic', ['#2c3e50', '#3498db'])
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Epigenetic State', color='#ecf0f1')
            cbar.ax.yaxis.set_tick_params(color='#ecf0f1')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ecf0f1')
            
            # Set labels
            ax.set_xlabel('Genes', color='#ecf0f1')
            ax.set_ylabel('Individuals', color='#ecf0f1')
            ax.set_title('Epigenetic Patterns', color='#ecf0f1')
            
            # Set tick colors
            ax.tick_params(axis='x', colors='#ecf0f1')
            ax.tick_params(axis='y', colors='#ecf0f1')
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_color('#ecf0f1')
            
            # Add to layout
            layout.addWidget(canvas)
            
            # Show window
            epi_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to visualize epigenetic patterns: {str(e)}")
    
    def export_plots(self):
        """Export plots to file with high resolution"""
        if not self.generation_data:
            self.status_bar.showMessage("No data to export!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if filename:
            try:
                # Update plots to ensure they're current
                self.update_plots()
                
                # Save with high DPI and tight layout
                self.canvas.figure.savefig(filename, dpi=300, facecolor='#2c3e50', bbox_inches='tight')
                self.status_bar.showMessage(f"Plot saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export plot: {str(e)}")
                self.status_bar.showMessage("Failed to export plot")
    
    def save_results(self):
        """Save simulation results to file with all data"""
        if not self.generation_data:
            self.status_bar.showMessage("No data to save!")
            return
            
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Results", "", 
            "JSON Files (*.json);;CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        # Ensure proper extension is added if not provided
        if filename:
            if selected_filter == "CSV Files (*.csv)" and not filename.endswith('.csv'):
                filename += '.csv'
            elif selected_filter == "JSON Files (*.json)" and not filename.endswith('.json'):
                filename += '.json'
            elif selected_filter == "Excel Files (*.xlsx)" and not filename.endswith('.xlsx'):
                filename += '.xlsx'
        
        if filename:
            try:
                # Prepare complete data dictionary with detailed parameters
                data = {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "application_version": "Classical Mating Kernel v1.0",
                        "simulation_duration": f"{len(self.generation_data)} generations"
                    },
                    "kernel_parameters": {
                        "symmetry_alpha": self.alpha_slider.findChild(QSlider).value() / 100.0,
                        "stochasticity_beta": self.beta_slider.findChild(QSlider).value() / 100.0,
                        "granularity_gamma": self.gamma_slider.findChild(QSlider).value() / 100.0,
                        "compatibility_delta": self.delta_slider.findChild(QSlider).value() / 100.0,
                        "fluke_contingency_epsilon": self.epsilon_slider.findChild(QSlider).value() / 100.0,
                        "speciation_temperature_zeta": self.zeta_slider.findChild(QSlider).value() / 100.0
                    },
                    "advanced_parameters": {
                        "neural_interface": self.neural_check.isChecked() if hasattr(self, 'neural_check') else False,
                        "topological_features": self.topology_check.isChecked() if hasattr(self, 'topology_check') else False,
                        "developmental_plasticity": self.plasticity_spin.findChild(QDoubleSpinBox).value() if hasattr(self, 'plasticity_spin') else 0.1,
                        "lineage_tracking": self.lineage_check.isChecked() if hasattr(self, 'lineage_check') else True,
                        "epigenetic_system": self.epigenetic_check.isChecked() if hasattr(self, 'epigenetic_check') else True,
                        "lineage_depth_limit": self.depth_limit_spin.findChild(QSpinBox).value() if hasattr(self, 'depth_limit_spin') else 50,
                        "population_floor": self.pop_floor_spin.findChild(QSpinBox).value() if hasattr(self, 'pop_floor_spin') else 20,
                        "archive_rate": self.archive_rate_spin.findChild(QDoubleSpinBox).value() if hasattr(self, 'archive_rate_spin') else 15,
                        "epigenetic_pulse_period": self.epi_pulse_spin.findChild(QSpinBox).value() if hasattr(self, 'epi_pulse_spin') else 10,
                        "sigma_reference": self.sigma_ref_spin.findChild(QDoubleSpinBox).value() if hasattr(self, 'sigma_ref_spin') else 1.0,
                        "tau_epsilon": self.tau_epsilon_spin.findChild(QDoubleSpinBox).value() if hasattr(self, 'tau_epsilon_spin') else 2.0
                    },
                    "population_parameters": {
                        "population_size": self.pop_size_spin.findChild(QSpinBox).value(),
                        "total_generations": self.generations_spin.findChild(QSpinBox).value(),
                        "genome_dimension": self.genome_dim_spin.findChild(QSpinBox).value(),
                        "mutation_rate": self.mutation_spin.findChild(QDoubleSpinBox).value(),
                        "simulation_speed": self.speed_spin.findChild(QDoubleSpinBox).value(),
                        "lineage_tracking_enabled": self.lineage_check.isChecked(),
                        "lineage_depth_limit": self.depth_limit_spin.findChild(QSpinBox).value(),
                        "epigenetic_system_enabled": self.epigenetic_check.isChecked()
                    },
                    "generation_data": self.generation_data,
                    "population_history": self.population_history
                }
                
                # Add lineage and epigenetic data if available
                if hasattr(self.simulation_thread, 'lineage_tree'):
                    data["lineage_data"] = {
                        "lineage_tree": dict(self.simulation_thread.lineage_tree),
                        "current_hashes": self.simulation_thread.lineage_hashes
                    }
                
                if hasattr(self.simulation_thread, 'epigenetic_tags'):
                    data["epigenetic_data"] = {
                        "epigenetic_tags": {k: v.tolist() for k, v in self.simulation_thread.epigenetic_tags.items()}
                    }
                
                # Save based on file type
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                elif filename.endswith('.csv'):
                    # Create combined DataFrame with parameters and metrics
                    metrics_df = pd.DataFrame(data["generation_data"])
                    
                    # Add ALL parameters as columns (repeated for each row)
                    for param_category in ["kernel_parameters", "population_parameters", "advanced_parameters"]:
                        for param, value in data[param_category].items():
                            if isinstance(value, dict):
                                for sub_param, sub_value in value.items():
                                    metrics_df[f"{param}.{sub_param}"] = sub_value
                            else:
                                metrics_df[param] = value
                    
                    metrics_df.to_csv(filename, index=False)
                elif filename.endswith('.xlsx'):
                    try:
                        import openpyxl
                        # Create Excel file with multiple sheets
                        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                            # Metrics sheet with ALL parameters included
                            metrics_df = pd.DataFrame(data["generation_data"])
                            for param_category in ["kernel_parameters", "population_parameters", "advanced_parameters"]:
                                for param, value in data[param_category].items():
                                    if isinstance(value, dict):
                                        for sub_param, sub_value in value.items():
                                            metrics_df[f"{param}.{sub_param}"] = sub_value
                                    else:
                                        metrics_df[param] = value
                            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                            
                            # Parameters sheet with detailed breakdown
                            params_data = []
                            for param_category in ["kernel_parameters", "population_parameters", "advanced_parameters"]:
                                for param, value in data[param_category].items():
                                    if isinstance(value, dict):
                                        for sub_param, sub_value in value.items():
                                            params_data.append({
                                                "Category": param_category,
                                                "Parameter": f"{param}.{sub_param}",
                                                "Value": sub_value
                                            })
                                    else:
                                        params_data.append({
                                            "Category": param_category,
                                            "Parameter": param,
                                            "Value": value
                                        })
                            
                            pd.DataFrame(params_data).to_excel(
                                writer, sheet_name='Parameters', index=False)
                            
                            # Metadata sheet
                            pd.DataFrame.from_dict(data["metadata"], orient='index').to_excel(
                                writer, sheet_name='Metadata')
                    except ImportError:
                        QMessageBox.critical(self, "Export Error", 
                            "Excel export requires openpyxl package. Please install with:\n\n"
                            "pip install openpyxl")
                        return
                
                self.status_bar.showMessage(f"Results saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")
                self.status_bar.showMessage("Failed to save results")
    
    def export_lineage_data(self):
        """Export lineage data to JSON file with complete metadata"""
        if not hasattr(self, 'simulation_thread') or not self.simulation_thread or not hasattr(self.simulation_thread, 'lineage_tree'):
            self.status_bar.showMessage("No lineage data to export!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Lineage Data", "", "JSON Files (*.json);;All Files (*)",
            options=QFileDialog.Options()
        )
        if filename and not filename.endswith('.json'):
            filename += '.json'
        
        if filename:
            try:
                # Prepare comprehensive lineage data
                lineage_data = {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "generation": self.simulation_thread.current_generation if hasattr(self.simulation_thread, 'current_generation') else 0,
                        "population_size": len(self.simulation_thread.lineage_hashes) if hasattr(self.simulation_thread, 'lineage_hashes') else 0
                    },
                    "lineage_tree": dict(self.simulation_thread.lineage_tree),
                    "current_hashes": self.simulation_thread.lineage_hashes,
                    "max_depth": self.calculate_lineage_depth_from_tree(self.simulation_thread.lineage_tree),
                    "statistics": {
                        "unique_lineages": len(set(self.simulation_thread.lineage_hashes.values())),
                        "average_branching": sum(len(v) for v in self.simulation_thread.lineage_tree.values())/max(1, len(self.simulation_thread.lineage_tree))
                    }
                }
                
                # Save with pretty printing and sorted keys
                with open(filename, 'w') as f:
                    json.dump(lineage_data, f, indent=2, sort_keys=True)
                
                self.status_bar.showMessage(f"Lineage data saved to {filename}")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export lineage data: {str(e)}")
                self.status_bar.showMessage("Failed to export lineage data")
                return False
    
    def export_epigenetic_data(self):
        """Export comprehensive epigenetic data to JSON file"""
        if not hasattr(self, 'simulation_thread') or not self.simulation_thread or not hasattr(self.simulation_thread, 'epigenetic_tags'):
            self.status_bar.showMessage("No epigenetic data to export!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Epigenetic Data", "", "JSON Files (*.json);;All Files (*)",
            options=QFileDialog.Options()
        )
        if filename and not filename.endswith('.json'):
            filename += '.json'
        
        if filename:
            try:
                # Prepare comprehensive epigenetic data
                epigenetic_data = {
                    "metadata": {
                        "export_time": datetime.now().isoformat(),
                        "generation": self.simulation_thread.current_generation if hasattr(self.simulation_thread, 'current_generation') else 0,
                        "population_size": len(self.simulation_thread.epigenetic_tags),
                        "genome_dim": len(next(iter(self.simulation_thread.epigenetic_tags.values()))) if self.simulation_thread.epigenetic_tags else 0
                    },
                    "epigenetic_tags": {k: v.tolist() for k, v in self.simulation_thread.epigenetic_tags.items()},
                    "statistics": {
                        "variation": self.generation_data[-1].get("epigenetic_variation", 0) if self.generation_data else 0,
                        "activation_rate": np.mean([np.mean(tags) for tags in self.simulation_thread.epigenetic_tags.values()]),
                        "spatial_correlation": self.calculate_epigenetic_spatial_correlation(),
                        "inheritance_patterns": self.analyze_epigenetic_inheritance()
                    },
                    "developmental_effects": {
                        "plasticity": self.plasticity_spin.findChild(QDoubleSpinBox).value(),
                        "environmental_modulation": self.analyze_environmental_modulation()
                    }
                }
                
                # Save with pretty printing and sorted keys
                with open(filename, 'w') as f:
                    json.dump(epigenetic_data, f, indent=2, sort_keys=True)
                
                self.status_bar.showMessage(f"Epigenetic data saved to {filename}")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export epigenetic data: {str(e)}")
                self.status_bar.showMessage("Failed to export epigenetic data")
                return False

    def calculate_epigenetic_spatial_correlation(self):
        """Calculate spatial correlation of epigenetic tags"""
        if not hasattr(self.simulation_thread, 'epigenetic_tags') or not self.simulation_thread.epigenetic_tags:
            return 0.0
        
        tags_matrix = np.array(list(self.simulation_thread.epigenetic_tags.values()))
        mean_tags = np.mean(tags_matrix, axis=0)
        return np.mean(np.abs(np.diff(mean_tags)))

    def analyze_epigenetic_inheritance(self):
        """Analyze epigenetic inheritance patterns"""
        if not hasattr(self.simulation_thread, 'lineage_tree') or not self.simulation_thread.lineage_tree:
            return {}
        
        inheritance_stats = {
            "parent_child_correlation": 0.0,
            "mutation_rate": 0.0,
            "conserved_regions": []
        }
        
        # Basic implementation - can be enhanced
        if len(self.simulation_thread.epigenetic_tags) > 1:
            tags = list(self.simulation_thread.epigenetic_tags.values())
            inheritance_stats["mutation_rate"] = np.mean([np.mean(tags[i] != tags[i+1]) 
                                                         for i in range(len(tags)-1)])
        
        return inheritance_stats

    def analyze_environmental_modulation(self):
        """Analyze environmental effects on epigenetic patterns"""
        if not self.generation_data:
            return {}
            
        return {
            "generation_correlation": np.corrcoef(
                [d["generation"] for d in self.generation_data],
                [d.get("epigenetic_variation", 0) for d in self.generation_data]
            )[0,1]
        }
    
    def export_population_data(self):
        """Export comprehensive population snapshots to file with metadata"""
        if not self.population_history:
            self.status_bar.showMessage("No population data to export!")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Population Data", "", 
            "JSON Files (*.json);;NPZ Files (*.npz);;CSV Files (*.csv)",
            options=QFileDialog.Options()
        )
        
        # Ensure proper extension is added if not provided
        if filename:
            if 'JSON Files' in self.sender().text() and not filename.endswith('.json'):
                filename += '.json'
            elif 'NPZ Files' in self.sender().text() and not filename.endswith('.npz'):
                filename += '.npz'
            elif 'CSV Files' in self.sender().text() and not filename.endswith('.csv'):
                filename += '.csv'
        
        if filename:
            try:
                if filename.endswith('.json'):
                    # Save as JSON with metadata
                    data = {
                        "metadata": {
                            "export_time": datetime.now().isoformat(),
                            "num_snapshots": len(self.population_history),
                            "generations": [i*10 for i in range(len(self.population_history))],
                            "population_size": len(self.population_history[0]) if self.population_history else 0,
                            "genome_dim": self.population_history[0].shape[1] if self.population_history else 0
                        },
                        "snapshots": [pop.tolist() for pop in self.population_history],
                        "statistics": {
                            "mean_diversity": [self.calculate_diversity(pop) for pop in self.population_history],
                            "mean_fitness": [np.mean(self.evaluate_fitness(pop)) for pop in self.population_history]
                        }
                    }
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                elif filename.endswith('.npz'):
                    # Save as compressed numpy array with metadata
                    np.savez_compressed(
                        filename,
                        *self.population_history,
                        generations=np.array([i*10 for i in range(len(self.population_history))]),
                        diversity=np.array([self.calculate_diversity(pop) for pop in self.population_history]))
                elif filename.endswith('.csv'):
                    # Save last generation as CSV with metadata header
                    df = pd.DataFrame(self.population_history[-1])
                    with open(filename, 'w') as f:
                        f.write(f"# Population snapshot from generation {10*(len(self.population_history)-1)}\n")
                        f.write(f"# Population size: {len(self.population_history[-1])}\n")
                        f.write(f"# Genome dimension: {self.population_history[-1].shape[1]}\n")
                        df.to_csv(f, index=False)
                
                self.status_bar.showMessage(f"Population data saved to {filename}")
                return True
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export population data: {str(e)}")
                self.status_bar.showMessage("Failed to export population data")
                return False
    
    def calculate_lineage_depth_from_tree(self, lineage_tree):
        """Calculate maximum lineage depth from lineage tree"""
        if not lineage_tree:
            return 0
        
        def get_depth(hash_val):
            if not lineage_tree.get(hash_val, []):
                return 1
            return 1 + max(get_depth(p) for p in lineage_tree[hash_val])
        
        return max(get_depth(h) for h in lineage_tree.keys())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create and show splash screen
    splash = SplashScreen()
    splash.show()
    
    # Process events to make sure splash is shown
    app.processEvents()
    
    # Create main application window (don't show yet)
    window = ClassicalMatingKernelApp()
    window.hide()  # Keep hidden until splash is done
    
    # Simulate loading time
    start_time = time.time()
    while time.time() - start_time < 3:  # Reduced from 10 to 3 seconds for better UX
        app.processEvents()
        time.sleep(0.1)
    
    # Close splash and show main window properly
    splash.finish(window)
    window.showMaximized()
    window.activateWindow()
    window.raise_()
    
    sys.exit(app.exec_())
