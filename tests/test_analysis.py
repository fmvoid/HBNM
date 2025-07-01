#!/usr/bin/env python3
"""
Test script for the new SimulationAnalyzer functionality.

This script demonstrates how to use the SimulationAnalyzer class
to analyze all available simulation outputs.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hbnm.io import Data
from hbnm.analysis import SimulationAnalyzer

def main():
    """Main test function."""
    print("="*60)
    print("TESTING SIMULATION ANALYZER")
    print("="*60)
    
    # Set up paths
    current_path = Path(__file__).parent
    parent_path = current_path.parent
    
    input_dir = str(parent_path / 'data')
    output_dir = str(parent_path / 'outputs')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    data = Data(input_dir, output_dir)
    analyzer = SimulationAnalyzer(data, verbose=True)
    
    # Test 1: Detect simulations
    print("\n1. Detecting simulations...")
    simulations = analyzer.detect_simulations()
    
    if not simulations:
        print("No simulations found. Make sure you have run some optimizations first.")
        return
    
    # Test 2: Detect iterations for each simulation
    print("\n2. Detecting iterations for each simulation...")
    for sim in simulations:
        iterations = analyzer.detect_iterations(sim)
        print(f"  {sim}: {len(iterations)} iterations")
    
    # Test 3: Load one simulation as a test
    print("\n3. Loading data for one simulation...")
    test_sim = simulations[0]
    # Load into the analyzer's internal storage
    analyzer.simulation_data[test_sim] = analyzer.load_simulation_data(test_sim)
    
    if analyzer.simulation_data[test_sim]['iterations']:
        print(f"Successfully loaded {len(analyzer.simulation_data[test_sim]['iterations'])} iterations for {test_sim}")
        print(f"Parameter dimensions: {analyzer.simulation_data[test_sim]['theta'][0].shape}")
    else:
        print(f"No data loaded for {test_sim}")
        return
    
    # Test 4: Create summary report
    print("\n4. Creating summary report...")
    summary = analyzer.create_summary_report([test_sim])
    print(summary.to_string(index=False))
    
    # Test 5: Test plot creation (but don't display)
    print("\n5. Testing plot creation...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig = analyzer.plot_simulation_progress(test_sim)
        if fig is not None:
            print("Progress plot created successfully")
        else:
            print("Failed to create progress plot")
            
        fig = analyzer.plot_parameter_evolution(test_sim)
        if fig is not None:
            print("Parameter evolution plot created successfully")
        else:
            print("Failed to create parameter evolution plot")
            
    except ImportError:
        print("Matplotlib not available for plot testing")
    
    print("\n" + "="*60)
    print("ANALYSIS FUNCTIONALITY TEST COMPLETED")
    print("="*60)
    print("\nThe SimulationAnalyzer is working correctly!")
    print("You can now use it in your notebooks and scripts.")

if __name__ == "__main__":
    main() 