def test_sample_from_harmonic_osciallator():
    # use local moves to sample from the harmonic oscillator
    from openmm import unit

    from openmmtools.testsystems import HarmonicOscillator

    ho = HarmonicOscillator()

    from chiron.potential import HarmonicOscillatorPotential

    # NOTE: let's construct this potential from the openmmtools test system ( dominic)
    harmonic_potential = HarmonicOscillatorPotential(ho.topology, ho.K, U0=ho.U0)
    from chiron.integrators import LangevinIntegrator

    integrator = LangevinIntegrator(
        stepsize=0.2 * unit.femtosecond,
    )

    r = integrator.run(
        ho.positions,
        harmonic_potential,
        temperature=300 * unit.kelvin,
        n_steps=5,
    )

    import jax.numpy as jnp

    reference_energy = jnp.array(
        [0.0, 0.00018982, 0.00076115, 0.00172312, 0.00307456, 0.00480607]
    )
    jnp.allclose(jnp.array(r["energy"]).flatten(), reference_energy)


def test_sample_from_harmonic_osciallator_with_MCMC_classes():
    # use local moves to sample from the HO, but use the MCMC classes
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

    # Initalize the move set (here only LangevinDynamicsMove)
    langevin_move = LangevinDynamicsMove()

    move_set = MoveSet({"LangevinDynamics": langevin_move}, [("LangevinDynamics", 1)])

    # Initalize the sampler
    sampler = GibbsSampler(move_set)
    
    # Run the sampler with the thermodynamic state and sampler state and return the sampler state
    sampler.run(thermodynamic_state, sampler_state, nr_of_repeats=2) # how many times to repeat


def test_sample_from_joint_distribution_of_two_HO_with_local_moves_and_MC_updates():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using local langevin moves
    # and global moves that change the spring constants and equilibrium positions
    pass


def test_sample_from_joint_distribution_of_two_HO_with_MC_moves():
    # define two harmonic oscillators with different spring constants and equilibrium positions
    # sample from the joint distribution of the two HO using metropolis hastings moves
    pass
