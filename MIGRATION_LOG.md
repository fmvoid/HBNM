# h-BNM Python 2 → 3 Migration Log

## Project Overview
- **Original**: Hierarchical Brain Network Model for computational neuroscience
- **Migration**: Python 2.7.12 → 3.9
- **Complexity**: Advanced - implements techniques from literature
- **Learning Goal**: Understand real-world modeling techniques while migrating

## File Structure Discovered
### Core Files (Priority 1):
- `hbnm/bnm.py` - Main wrapper class
- `hbnm/model/dmf.py` - Core brain model (dynamic mean field)
- `hbnm/pmc.py` - Population Monte Carlo optimization

### Supporting Files (Priority 2):
- `hbnm/model/sim.py` - Simulation engine
- `hbnm/model/utils.py` - Utility functions
- `hbnm/model/hemo.py` - Hemodynamic response function
- `hbnm/io.py` - Data I/O

### Parameter Sets:
- `hbnm/model/params/` - Literature parameter sets

### Example Scripts:
- `scripts/` - Optimization examples, homogeneous vs heterogeneous models

## Migration Strategy
1. Start with core imports and basic syntax
2. Focus on getting main classes working
3. Learn the neuroscience as we fix the code
4. Test with simple examples

### Potential changes
- What's the best place for the `load_data` function