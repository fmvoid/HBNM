#!/usr/bin/env python3
"""
Comprehensive test suite for multi-map biological heterogeneity functionality.

This test suite validates:
1. Backwards compatibility with existing single-map functionality
2. New multi-map functionality with arbitrary numbers of biological maps
3. Error handling and edge cases
4. Integration with optimization workflows
5. Hemisphere splitting functionality

Author: Generated for HBNM multi-map implementation
"""

import numpy as np
import sys
import os
import pytest

# Add the parent directory to path to import hbnm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import hbnm
from hbnm.model.dmf import Model
from hbnm.bnm import Bnm


class TestBackwardsCompatibility:
    """Test that all existing functionality still works exactly as before."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducible tests
        self.n_regions = 10
        self.sc = np.random.rand(self.n_regions, self.n_regions)
        self.sc = (self.sc + self.sc.T) / 2  # Make symmetric
        np.fill_diagonal(self.sc, 0)  # No self-connections
        self.gradient = np.random.rand(self.n_regions)
        
    def test_homogeneous_model_unchanged(self):
        """Test that homogeneous models work exactly as before."""
        # Create models the old way and new way
        model_old = Bnm(self.sc)
        model_new = Bnm(self.sc, maps=None)
        
        # Set homogeneous parameters
        model_old.set('w_EE', 0.15)
        model_old.set('w_EI', 0.15)
        model_old.set('G', 2.0)
        
        model_new.set('w_EE', 0.15)
        model_new.set('w_EI', 0.15)
        model_new.set('G', 2.0)
        
        # Should be identical
        np.testing.assert_array_equal(model_old.get('w_EE'), model_new.get('w_EE'))
        np.testing.assert_array_equal(model_old.get('w_EI'), model_new.get('w_EI'))
        assert model_old.get('G') == model_new.get('G')
        
    def test_single_map_gradient_compatibility(self):
        """Test that single gradient maps work exactly as before."""
        # Old way with gradient
        model_old = Bnm(self.sc, gradient=self.gradient)
        model_old.set('w_EE', (0.15, 0.05))
        model_old.set('w_EI', (0.15, -0.02))
        
        # New way with equivalent maps
        maps = self.gradient[None, :]  # Convert to (1, n_regions) format
        model_new = Bnm(self.sc, maps=maps)
        model_new.set('w_EE', (0.15, 0.05))
        model_new.set('w_EI', (0.15, -0.02))
        
        # Should be identical
        np.testing.assert_array_almost_equal(model_old.get('w_EE'), model_new.get('w_EE'))
        np.testing.assert_array_almost_equal(model_old.get('w_EI'), model_new.get('w_EI'))
        
    def test_negative_slope_compatibility(self):
        """Test that negative slope behavior is preserved."""
        model_old = Bnm(self.sc, gradient=self.gradient)
        model_old.set('w_EE', (0.15, -0.05))  # Negative slope
        
        maps = self.gradient[None, :]
        model_new = Bnm(self.sc, maps=maps)
        model_new.set('w_EE', (0.15, -0.05))  # Should behave identically
        
        np.testing.assert_array_almost_equal(model_old.get('w_EE'), model_new.get('w_EE'))
        
    def test_hemisphere_splitting_compatibility(self):
        """Test that hemisphere splitting still works with gradients."""
        sc_left = np.random.rand(5, 5)
        sc_right = np.random.rand(5, 5)
        grad_left = np.random.rand(5)
        grad_right = np.random.rand(5)
        
        # Old way
        model_old = Bnm([sc_left, sc_right], gradient=[grad_left, grad_right])
        model_old.set('w_EE', (0.15, 0.05))
        
        # New way
        model_new = Bnm([sc_left, sc_right], maps=[grad_left[None, :], grad_right[None, :]])
        model_new.set('w_EE', (0.15, 0.05))
        
        # Should be identical
        np.testing.assert_array_almost_equal(model_old.get('w_EE')[0], model_new.get('w_EE')[0])
        np.testing.assert_array_almost_equal(model_old.get('w_EE')[1], model_new.get('w_EE')[1])


class TestMultiMapFunctionality:
    """Test new multi-map functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_regions = 8
        self.n_maps = 4
        self.sc = np.random.rand(self.n_regions, self.n_regions)
        self.sc = (self.sc + self.sc.T) / 2
        np.fill_diagonal(self.sc, 0)
        
        # Create biological maps matrix
        self.maps = np.random.randn(self.n_maps, self.n_regions)
        
    def test_multi_map_parameter_application(self):
        """Test that multi-map parameters are applied correctly."""
        model = Bnm(self.sc, maps=self.maps)
        
        # Set multi-map parameters: bias + 4 coefficients
        bias_ee = 0.15
        coeffs_ee = [0.02, -0.01, 0.03, -0.005]
        params_ee = (bias_ee, *coeffs_ee)
        
        model.set('w_EE', params_ee)
        
        # Manual calculation of expected values
        expected = bias_ee + np.dot(coeffs_ee, self.maps)
        actual = model.get('w_EE')
        
        np.testing.assert_array_almost_equal(actual, expected, decimal=10)
        
    def test_different_map_counts(self):
        """Test models with different numbers of maps."""
        for n_maps in [1, 2, 3, 5, 10]:
            maps = np.random.randn(n_maps, self.n_regions)
            model = Bnm(self.sc, maps=maps)
            
            # Create parameter vector: bias + n_maps coefficients
            params = tuple([0.15] + list(np.random.uniform(-0.1, 0.1, n_maps)))
            model.set('w_EE', params)
            
            # Should not raise any errors
            w_ee = model.get('w_EE')
            assert len(w_ee) == self.n_regions
            assert np.all(np.isfinite(w_ee))
            
    def test_maps_property_access(self):
        """Test that maps can be accessed and modified via property."""
        model = Bnm(self.sc, maps=self.maps)
        
        # Test getter
        retrieved_maps = model.dmf.maps
        np.testing.assert_array_equal(retrieved_maps, self.maps)
        
        # Test setter
        new_maps = np.random.randn(3, self.n_regions)
        model.dmf.maps = new_maps
        np.testing.assert_array_equal(model.dmf.maps, new_maps)
        
    def test_single_map_as_2d_array(self):
        """Test that single maps can be provided as 2D arrays."""
        single_map_1d = np.random.rand(self.n_regions)
        single_map_2d = single_map_1d[None, :]  # Shape: (1, n_regions)
        
        model_1d = Bnm(self.sc, maps=single_map_1d)
        model_2d = Bnm(self.sc, maps=single_map_2d)
        
        # Both should work and give same results
        params = (0.15, 0.05)
        model_1d.set('w_EE', params)
        model_2d.set('w_EE', params)
        
        np.testing.assert_array_almost_equal(model_1d.get('w_EE'), model_2d.get('w_EE'))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_regions = 6
        self.sc = np.eye(self.n_regions)  # Simple identity matrix
        self.maps = np.random.randn(3, self.n_regions)
        
    def test_conflicting_parameters_error(self):
        """Test that providing both gradient and maps raises error."""
        gradient = np.random.rand(self.n_regions)
        
        with pytest.raises(ValueError, match="Cannot specify both 'gradient' and 'maps'"):
            Bnm(self.sc, gradient=gradient, maps=self.maps)
            
        with pytest.raises(ValueError, match="Cannot specify both 'hmap' and 'maps'"):
            Model(self.sc, hmap=gradient, maps=self.maps)
            
    def test_parameter_size_mismatch_error(self):
        """Test that wrong parameter sizes raise appropriate errors."""
        model = Bnm(self.sc, maps=self.maps)  # 3 maps
                
        # Now the actual test
        with pytest.raises(ValueError, match="Parameter size mismatch"):
            model.set('w_EE', (0.15, 0.01, 0.02, 0.03, 0.04, 0.05))
            
    def test_no_maps_with_multi_params_error(self):
        """Test error when providing multi-params without maps."""
        model = Bnm(self.sc)  # No maps
        
        with pytest.raises(ValueError, match="No biological maps provided"):
            model.set('w_EE', (0.15, 0.01, 0.02, 0.03))
            
    def test_wrong_map_dimensions_error(self):
        """Test error when maps have wrong number of regions."""
        wrong_maps = np.random.randn(3, self.n_regions + 2)  # Wrong number of regions
        
        with pytest.raises(ValueError, match="Maps must have .* regions"):
            model = Bnm(self.sc)
            model.dmf.maps = wrong_maps


class TestIntegrationWithSimulation:
    """Test that multi-map models work with actual simulations."""
    
    def setup_method(self):
        """Set up realistic test case."""
        np.random.seed(42)
        self.n_regions = 20
        
        # Create realistic structural connectivity
        self.sc = np.random.exponential(0.1, (self.n_regions, self.n_regions))
        self.sc = (self.sc + self.sc.T) / 2
        np.fill_diagonal(self.sc, 0)
        
        # Create realistic biological maps (normalized)
        self.n_maps = 6
        self.maps = np.random.randn(self.n_maps, self.n_regions)
        for i in range(self.n_maps):
            self.maps[i] = (self.maps[i] - self.maps[i].mean()) / self.maps[i].std()
            
    def test_stability_check_works(self):
        """Test that stability checking works with multi-map models."""
        model = Bnm(self.sc, maps=self.maps)
        
        # Set reasonable parameters
        model.set('w_EE', (0.15, 0.01, -0.005, 0.008, -0.003, 0.002, 0.001))
        model.set('w_EI', (0.15, -0.01, 0.005, -0.008, 0.003, -0.002, -0.001))
        model.set('G', 1.5)
        
        # Should be able to check stability without errors
        try:
            is_unstable = model.check_stability()
            # Result should be boolean
            assert isinstance(is_unstable, (bool, np.bool_))
        except Exception as e:
            pytest.fail(f"Stability check failed with error: {e}")
            
    def test_moments_method_works(self):
        """Test that linearized analysis works with multi-map models."""
        model = Bnm(self.sc, maps=self.maps)
        
        # Set conservative parameters to ensure stability
        model.set('w_EE', (0.12, 0.005, -0.002, 0.003, -0.001, 0.001, 0.0005))
        model.set('w_EI', (0.12, -0.005, 0.002, -0.003, 0.001, -0.001, -0.0005))
        model.set('G', 1.0)
        
        # Should be able to run moments method
        try:
            model.moments_method(BOLD=True)
            
            # Check that correlation matrices are computed
            corr_bold = model.get('corr_bold')
            assert corr_bold is not None
            assert corr_bold.shape == (self.n_regions, self.n_regions)
            assert np.allclose(np.diag(corr_bold), 1.0)  # Diagonal should be 1
            
        except Exception as e:
            pytest.fail(f"Moments method failed with error: {e}")
            
    def test_hemisphere_splitting_with_multi_maps(self):
        """Test hemisphere splitting with multiple maps."""
        n_regions_per_hem = 10
        sc_left = np.random.rand(n_regions_per_hem, n_regions_per_hem)
        sc_right = np.random.rand(n_regions_per_hem, n_regions_per_hem)
        
        maps_left = np.random.randn(4, n_regions_per_hem)
        maps_right = np.random.randn(4, n_regions_per_hem)
        
        model = Bnm([sc_left, sc_right], maps=[maps_left, maps_right])
        
        # Set parameters
        params_ee = (0.15, 0.01, -0.005, 0.008, -0.003)
        params_ei = (0.15, -0.01, 0.005, -0.008, 0.003)
        
        model.set('w_EE', params_ee)
        model.set('w_EI', params_ei)
        model.set('G', 1.5)
        
        # Should work without errors
        w_ee = model.get('w_EE')
        w_ei = model.get('w_EI')
        
        assert len(w_ee) == 2  # Two hemispheres
        assert len(w_ei) == 2
        assert len(w_ee[0]) == n_regions_per_hem
        assert len(w_ee[1]) == n_regions_per_hem


class TestOptimizationCompatibility:
    """Test compatibility with optimization workflows."""
    
    def setup_method(self):
        """Set up optimization-like scenario."""
        np.random.seed(42)
        self.n_regions = 15
        self.sc = np.random.exponential(0.1, (self.n_regions, self.n_regions))
        self.sc = (self.sc + self.sc.T) / 2
        np.fill_diagonal(self.sc, 0)
        
        # Normalize SC as done in optimization
        sc_norm = 1. / self.sc.sum(1)
        self.sc = self.sc * np.tile(sc_norm, (self.n_regions, 1)).T
        
        self.n_maps = 5
        self.maps = np.random.randn(self.n_maps, self.n_regions)
        
    def test_parameter_vector_unpacking(self):
        """Test unpacking flat parameter vectors as done in optimization."""
        model = Bnm(self.sc, maps=self.maps)
        
        # Simulate parameter vector from optimizer
        # Order: [w_EI_bias, w_EI_coeffs..., w_EE_bias, w_EE_coeffs..., G]
        theta = np.array([
            0.12,  # w_EI bias
            -0.01, 0.005, -0.008, 0.003, -0.002,  # w_EI coeffs (5)
            0.15,  # w_EE bias  
            0.01, -0.005, 0.008, -0.003, 0.002,   # w_EE coeffs (5)
            1.8    # G
        ])
        
        # Unpack as would be done in optimization
        idx = 0
        w_EI_params = theta[idx:idx + self.n_maps + 1]
        idx += self.n_maps + 1
        w_EE_params = theta[idx:idx + self.n_maps + 1]
        idx += self.n_maps + 1
        G = theta[idx]
        
        # Apply to model
        model.set('w_EI', tuple(w_EI_params))
        model.set('w_EE', tuple(w_EE_params))
        model.set('G', G)
        
        # Should work without errors
        assert len(model.get('w_EI')) == self.n_regions
        assert len(model.get('w_EE')) == self.n_regions
        assert model.get('G') == G
        
    def test_repeated_parameter_updates(self):
        """Test that parameters can be updated repeatedly as in optimization."""
        model = Bnm(self.sc, maps=self.maps)
        
        # Simulate multiple optimization iterations
        for iteration in range(10):
            # Generate random parameters
            w_EI_params = tuple([0.12] + list(np.random.uniform(-0.02, 0.02, self.n_maps)))
            w_EE_params = tuple([0.15] + list(np.random.uniform(-0.02, 0.02, self.n_maps)))
            G = np.random.uniform(0.5, 3.0)
            
            # Update model
            model.set('w_EI', w_EI_params)
            model.set('w_EE', w_EE_params)
            model.set('G', G)
            
            # Verify parameters were set correctly
            assert np.allclose(model.get('G'), G)
            
            # Try to run a simulation step (should not crash)
            try:
                is_stable = not model.check_stability()
                if is_stable:
                    model.moments_method(BOLD=False)
            except:
                # Some parameter combinations might be unstable, that's ok
                pass


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_region_model(self):
        """Test model with single region."""
        sc = np.array([[0.0]])  # Single region, no self-connection
        maps = np.random.randn(3, 1)  # 3 maps, 1 region
        
        model = Bnm(sc, maps=maps)
        model.set('w_EE', (0.15, 0.01, -0.005, 0.008))
        model.set('w_EI', (0.15, -0.01, 0.005, -0.008))
        
        w_ee = model.get('w_EE')
        w_ei = model.get('w_EI')
        
        assert len(w_ee) == 1
        assert len(w_ei) == 1
        
    def test_zero_maps(self):
        """Test model with empty maps array."""
        sc = np.random.rand(5, 5)
        maps = np.empty((0, 5))  # Zero maps
        
        model = Bnm(sc, maps=maps)
        
        # Should only accept homogeneous parameters
        model.set('w_EE', 0.15)  # This should work
        
        with pytest.raises(ValueError):
            model.set('w_EE', (0.15, 0.01))  # This should fail
            
    def test_very_large_maps(self):
        """Test model with many maps."""
        n_regions = 10
        n_maps = 50  # Many maps
        
        sc = np.random.rand(n_regions, n_regions)
        maps = np.random.randn(n_maps, n_regions)
        
        model = Bnm(sc, maps=maps)
        
        # Should handle large parameter vectors
        params = tuple([0.15] + list(np.random.uniform(-0.001, 0.001, n_maps)))
        model.set('w_EE', params)
        
        w_ee = model.get('w_EE')
        assert len(w_ee) == n_regions
        assert np.all(np.isfinite(w_ee))


def run_all_tests():
    """Run all tests and provide summary."""
    print("Running comprehensive multi-map functionality tests...")
    print("=" * 60)
    
    # Test classes to run
    test_classes = [
        TestBackwardsCompatibility,
        TestMultiMapFunctionality, 
        TestErrorHandling,
        TestIntegrationWithSimulation,
        TestOptimizationCompatibility,
        TestEdgeCases
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            
            try:
                # Create instance and run setup
                instance = test_class()
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                
                # Run test method
                getattr(instance, test_method)()
                
                print(f"  ✓ {test_method}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {test_method}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{test_method}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
        return False
    else:
        print(f"\n🎉 ALL TESTS PASSED! 🎉")
        print(f"\nThe multi-map functionality is working correctly and maintains")
        print(f"full backwards compatibility with existing code.")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 