"""
Quadrotor Simulation Main Program.

This module serves as the entry point for running quadrotor simulations.
It handles parameter configuration and initializes the simulation environment.
"""

import argparse
import time

from simulation.plant import QuadrotorPlant
from simulation.controller import QuadrotorSMCController
from simulation.sim import QuadrotorSimulator


def main():
    """Main function to run the quadrotor simulation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Quadrotor UAV Simulation')
    parser.add_argument(
        '--initial', type=float, nargs=3, default=[0, 0, 0],
        help='Initial Position [x y z] (Default: [0 0 0])'
    )
    parser.add_argument(
        '--target', type=float, nargs=3, default=[1, 1, 2],
        help='Target Position [x y z] (Default: [1 1 2])'
    )
    parser.add_argument(
        '--time', type=float, default=10,
        help='Simulation duration (s) (Default: 10.0)'
    )
    parser.add_argument(
        '--dt', type=float, default=0.05,
        help='Time step (s) (Default: 0.05)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate static plots only (no animation)'
    )
    parser.add_argument(
        '--initial_attitude', type=float, nargs=3, default=[0, 0.2, 0.1],
        help='Initial Attitude [roll pitch yaw] (Default: [0 0.1 0.1]rad)'
    )
    
    args = parser.parse_args()
    
    # Create plant and controller
    plant = QuadrotorPlant()
    controller = QuadrotorSMCController(plant)
    
    # Create and configure simulator
    simulator = QuadrotorSimulator(
        plant=plant,
        controller=controller,
        initial_position=args.initial,
        target_position=args.target,
        initial_attitude=args.initial_attitude
    )
    simulator.dt = args.dt
    simulator.max_time = args.time
    
    # Print simulation parameters
    print(f"Running simulation:")
    print(f"  Initial position: {args.initial}")
    print(f"  Target position: {args.target}")
    print(f"  Initial attitude: {args.initial_attitude}")
    print(f"  Duration: {args.time} seconds")
    print(f"  Time step: {args.dt} seconds")
    
    # Run simulation and measure execution time
    start_time = time.time()
    simulator.run()
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Generate visualization
    if args.plot:
        simulator.plot_results()
    else:
        simulator.animate()


if __name__ == "__main__":
    main()