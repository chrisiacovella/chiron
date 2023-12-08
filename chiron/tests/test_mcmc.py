def test_sample_from_harmonic_osciallator(remove_h5_file):
    """
    Test sampling from a harmonic oscillator using local moves.

    This test initializes a harmonic oscillator from openmmtools.testsystems,
    sets up a harmonic potential, and uses a Langevin integrator to sample
    from the oscillator's state space.
    """
    from openmm import unit

    # initialize openmmtestsystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    # initialze HO potential
    from chiron.potential import HarmonicOscillatorPotential

    # NOTE: let's construct this potential from the openmmtools test system in the future
    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    # intialize the states
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        potential=harmonic_potential, temperature=300 * unit.kelvin
    )
    sampler_state = SamplerState(x0=ho.positions)
    from chiron.integrators import LangevinIntegrator

    from chiron.reporters import SimulationReporter

    reporter = SimulationReporter("test.h5", 1)

    integrator = LangevinIntegrator(
        stepsize=0.2 * unit.femtosecond, reporter=reporter, save_frequency=1
    )

    r = integrator.run(
        sampler_state,
        thermodynamic_state,
        n_steps=5,
    )

    import jax.numpy as jnp
    import h5py

    h5_file = "test.h5"
    h5 = h5py.File(h5_file, "r")
    keys = h5.keys()

    assert "energy" in keys, "Energy not in keys"
    assert "step" in keys, "Step not in keys"
    assert "traj" in keys, "Traj not in keys"

    energy = h5["energy"][:]
    print(energy)

    reference_energy = jnp.array(
        [0.00019308, 0.00077772, 0.00174247, 0.00307798, 0.00479007]
    )
    assert jnp.allclose(energy, reference_energy)


def test_sample_from_harmonic_osciallator_with_MCMC_classes_and_LangevinDynamics(
    remove_h5_file,
):
    """
    Test sampling from a harmonic oscillator using MCMC classes and Langevin dynamics.

    This test initializes a harmonic oscillator, sets up the thermodynamic and
    sampler states, and uses the Langevin dynamics move in an MCMC sampling scheme.
    """
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from chiron.mcmc import LangevinDynamicsMove, MoveSet, GibbsSampler

    # Initalize the testsystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential, temperature=300, volume=30 * (unit.angstrom**3)
    )
    sampler_state = SamplerState(ho.positions)

    # Initalize the move set (here only LangevinDynamicsMove) and reporter
    from chiron.reporters import SimulationReporter

    simulation_reporter = SimulationReporter("test.h5", 1)
    langevin_move = LangevinDynamicsMove(
        nr_of_steps=1_000, seed=1234, simulation_reporter=simulation_reporter
    )

    move_set = MoveSet([("LangevinMove", langevin_move)])

    # Initalize the sampler
    sampler = GibbsSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=2)  # how many times to repeat


def test_sample_from_harmonic_osciallator_with_MCMC_classes_and_MetropolisDisplacementMove(
    remove_h5_file,
):
    """
    Test sampling from a harmonic oscillator using MCMC classes and Metropolis displacement move.

    This test initializes a harmonic oscillator, sets up the thermodynamic and
    sampler states, and uses the Metropolis displacement move in an MCMC sampling scheme.
    """
    from openmm import unit
    from chiron.potential import HarmonicOscillatorPotential
    from chiron.mcmc import MetropolisDisplacementMove, MoveSet, GibbsSampler

    # Initalize the testsystem
    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential, temperature=300, volume=30 * (unit.angstrom**3)
    )
    sampler_state = SamplerState(ho.positions)

    # Initalize the move set and reporter
    from chiron.reporters import SimulationReporter

    simulation_reporter = SimulationReporter("test.h5", 1)

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=10,
        displacement_sigma=0.1 * unit.angstrom,
        atom_subset=[0],
        simulation_reporter=simulation_reporter,
    )

    move_set = MoveSet([("MetropolisDisplacementMove", mc_displacement_move)])

    # Initalize the sampler
    sampler = GibbsSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=2)  # how many times to repeat


def test_sample_from_harmonic_osciallator_array_with_MCMC_classes_and_MetropolisDisplacementMove(
    remove_h5_file,
):
    """
    Test sampling from a harmonic oscillator using MCMC classes and Metropolis displacement move.

    This test initializes a harmonic oscillator, sets up the thermodynamic and
    sampler states, and uses the Metropolis displacement move in an MCMC sampling scheme.
    """
    from openmm import unit
    from chiron.mcmc import MetropolisDisplacementMove, MoveSet, GibbsSampler

    # Initalize the testsystem
    from openmmtools.testsystems import HarmonicOscillatorArray

    ho = HarmonicOscillatorArray()

    # Initalize the potential
    from chiron.potential import HarmonicOscillatorPotential

    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K)

    # Initalize the sampler and thermodynamic state
    from chiron.states import ThermodynamicState, SamplerState

    thermodynamic_state = ThermodynamicState(
        harmonic_potential, temperature=300, volume=30 * (unit.angstrom**3)
    )
    sampler_state = SamplerState(ho.positions)

    # Initalize the move set and reporter
    from chiron.reporters import SimulationReporter

    simulation_reporter = SimulationReporter("test.h5", 1)

    mc_displacement_move = MetropolisDisplacementMove(
        nr_of_moves=10,
        displacement_sigma=0.1 * unit.angstrom,
        atom_subset=None,
        simulation_reporter=simulation_reporter,
    )

    move_set = MoveSet([("MetropolisDisplacementMove", mc_displacement_move)])

    # Initalize the sampler
    sampler = GibbsSampler(move_set, sampler_state, thermodynamic_state)

    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(n_iterations=2)  # how many times to repeat


def test_sample_from_joint_distribution_of_two_HO_with_local_moves_and_MC_updates():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using local langevin moves
    # and global moves that change the spring constants and equilibrium positions
    pass


def test_sample_from_joint_distribution_of_two_HO_with_MC_moves():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using metropolis hastings moves
    pass
