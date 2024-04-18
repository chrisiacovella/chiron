from __future__ import annotations
from openff.units import unit

# from openmm.app import Topology
from typing import Dict, List, Set, TYPE_CHECKING

import jax.numpy as jnp


# the topology class can be constructed from an openmm topology object
# the topology class needs to be able to perceive bonds, amino acids, ligands, waters, etc.
# the topology class needs to be able to perceive the center of mass of the system


if TYPE_CHECKING:
    from openmm.app.topology import Topology


class SystemTopology:
    def __init__(self) -> None:
        # set up the periodic table info
        from openff.units import elements

        # dictionary of atomic symbols with atomic number as the key
        self._pt_symbols = elements.SYMBOLS

        # dictionary of atomic masses with atomic number as the key
        self._pt_masses = elements.MASSES

        # generate a dictionary of atomic numbers with atomic symbol as the key
        self._pt_atomic_numbers = {}

        for atomic_number, atomic_symbol in self._pt_symbols.items():
            self._pt_atomic_numbers[atomic_symbol] = atomic_number

        # these system properties will be stored as jax.numpy arrays
        self._masses = None
        self._atomic_numbers = None
        self._connections = None

        # we will store all masses as amu, i.e., g/mol
        self._masses_unit = unit.g / unit.mol

        # the number of atoms in the system
        self._number_of_atoms = None

        # since "molecules" (i.e., connected atoms) are not necessarily of the same size
        # we will store the connected atoms as a list of lists
        # to use with jax we will need to pad the lists to the same size
        # the connected atoms will be used in routines such as a molecular MC barostat move
        # such that positions of connected atoms are moved together based on the center-of-mass
        # to maintain the molecular geometry
        self._connected_atoms = []

    @property
    def masses(self) -> jnp.array:
        if self._masses is None:
            return None
        # internally we will store masses as amu, but as jax arrays without units
        # we will attach units when we need to use them
        return self._masses * self._masses_unit

    @property
    def masses_unit(self) -> unit:
        """
        Returns the unit of the masses of the atoms in the system.

        Returns
        -------
        unit
            The unit of the masses of the atoms in the system.
        """

        return self._masses_unit

    @property
    def atomic_numbers(self) -> jnp.array:
        if self._atomic_numbers is None:
            return None
        return self._atomic_numbers

    @property
    def atomic_symbols(self) -> List[str]:
        """
        Returns the atomic symbols of the atoms in the system as a list

        Note, this is returned as a list of strings, as strings are not
        a supported dtype for jax arrays.

        Returns
        -------
        List[str]
            The atomic symbols of the atoms in the system.

        """
        if self._atomic_numbers is None:
            return None
        return [
            self.get_element(int(self._atomic_numbers[i]))
            for i in range(self._atomic_numbers.shape[0])
        ]

    @property
    def connections(self) -> jnp.array:
        """
        Returns the connections between atoms in the system.

        Returns
        -------
        jnp.array
            The connections between atoms in the system, of shape (n_connections, 2).
        """
        return self._connections

    @property
    def connected_atoms(self) -> List[List[int]]:
        """
        Returns the indices of "molecules" in the system, based upon the connections.

        Note, this only returns the list of connected atoms, it does not actually perform the calculation.
        Use the function `determine_connected_atoms()` to calculate the molecules.

        Note this is

        connected atoms.  This is done
        This iuses networkx to determine the molecules.  Note, this list also includes all single atom "molecules"
        i.e., those that are not connected to any other atom.

        Returns
        -------
        List[List[int]]
            The indices connected atoms in the system.
        """

        if self._connected_atoms is None:
            return None
        return self._connected_atoms

    def from_openmm(self, topology: Topology) -> None:
        """
        Convert an openmm topology object to a Chiron SystemTopology object.
        Parameters
        ----------
        topology

        Returns
        -------

        """
        from openmm.app.topology import Topology

        if isinstance(topology, Topology):
            from openff.units import openmm as off_openmm

            masses_list = []
            atomic_numbers_list = []

            for atom in topology.atoms():
                masses_list.append(
                    off_openmm.from_openmm(atom.element.mass).to(self._masses_unit).m
                )
                atomic_numbers_list.append(atom.element.atomic_number)

            self._masses = jnp.array(masses_list)
            self._atomic_numbers = jnp.array(atomic_numbers_list)
            self._number_of_atoms = self._atomic_numbers.shape[0]

            # add in code to extract connections
            for bond in topology.bonds():
                self.add_connection(bond.atom1.index, bond.atom2.index)
        else:
            raise ValueError(
                f"The input is not an openmm topology object. Got {type(topology)}"
            )

    def get_element(self, atomic_number: int) -> str:
        """
        Get the atomic symbol of an atom based on the atomic number.

        Parameters
        ----------
        atomic_number, int, required
            The atomic number of the atom of interest.

        Returns
        -------
        str
            The atomic symbol of the atom.
        """
        return self._pt_symbols[atomic_number]

    def remove_atom(self, atom_index) -> None:
        """
        Remove an atom from the system at a specific index.

        This will remove the atom from the system, as well as any connections that contain the atom.

        Note, removing an atom will change the atomic indices of the entire system
        Parameters
        ----------
        atom_index,
            The index of the atom to remove.

        """
        # remove an atom from the system at a specific index

        self._masses = jnp.delete(self._masses, atom_index)
        self._atomic_numbers = jnp.delete(self._atomic_numbers, atom_index)
        self._number_of_atoms = self._atomic_numbers.shape[0]
        # add in code to remove connections containing atom

    def add_atom(self, atom: Union[str, int]) -> None:
        """
        Add an individual atom to the system based on the atomic symbol or atomic number.

        This will be appended to the array of existing atoms in the system.

        Parameters
        ----------
        atom

        Returns
        -------

        """

        if isinstance(atom, str):
            atomic_number = self._pt_atomic_numbers[atom]
        elif isinstance(atom, int):
            atomic_number = atom
        else:
            raise ValueError(
                f"Atom must be either a string or an integer. Got {type(atom)}"
            )

        if self.masses is None:
            # masses in element are daltons which is equivalent to g/mol and amu
            # the g/mol unit is likely more useful for the masses
            self._masses = jnp.array([self._pt_masses[atomic_number].m])
            self._atomic_numbers = jnp.array([atomic_number])

        else:
            self._masses = jnp.append(self._masses, self._pt_masses[atomic_number].m)
            self._atomic_numbers = jnp.append(
                self._atomic_numbers, jnp.array([atomic_number])
            )
        self._number_of_atoms = self._atomic_numbers.shape[0]

    def add_atoms(self, atoms: List[Union[str, int]]) -> None:
        """
        Append multiple atoms to the system, passed as a list.  This will append to the existing atoms in the system.

        Parameters
        ----------
        atoms: List[str, int], required
            List of atoms to add to the system, either as atomic symbols or atomic numbers.

        """

        # setting these all at once is probably more efficient
        # but this routine is not going to be called frequently so this should be totally fine
        # and minimizes extra code
        for atom in atoms:
            self.add_atom(atom)

    def clear(self):
        """
        Reset all the system properties to None.

        """
        self.__init__()

    def add_connection(self, atom_index1: int, atom_index2: int) -> None:
        """
        Add a connection between two atoms in the system.

        Parameters
        ----------
        atom_index1, int, required
            The index of the first atom to connect.
        atom_index2
            The index of the second atom to connect.

        """
        # add a connection between two atoms

        if self._connections is None:
            self._connections = jnp.array([[atom_index1, atom_index2]])
        else:
            self._connections = jnp.append(
                self._connections, jnp.array([[atom_index1, atom_index2]]), axis=0
            )

    def add_connections(self, connections: List[List[int]]) -> None:
        """
        Add multiple connections to the system, passed as a list of lists.

        Parameters
        ----------
        connections: List[List[int]], required
            List of connections to add to the system, passed as a list of lists.

        """

        # setting these all at once is probably more efficient
        # but this routine is not going to be called frequently so this should be totally fine
        for connection in connections:
            self.add_connection(connection[0], connection[1])

    def determine_connected_atom(self):
        # determine the connected atoms of the system

        # if we have no connections, then each atom is a molecule
        if self._connections is None or self._connections.shape[0] == 0:
            for i in range(self._atomic_numbers.shape[0]):
                self._connected_atoms.append([i])
        else:
            self._connected_atoms = []
            import networkx as nx

            G = nx.Graph()

            for i in range(self._connections.shape[0]):
                connection = self._connections[i]
                G.add_edge(int(connection[0]), int(connection[1]))

            for comp in nx.connected_components(G):
                self._connected_atoms.append(list(comp))

            # calculate all the individual atoms that are not connected to any other atom

            atomic_indices = jnp.array(range(self._atomic_numbers.shape[0]))

            dif1 = jnp.setdiff1d(self._connections, atomic_indices)
            dif2 = jnp.setdiff1d(atomic_indices, self._connections)

            single_atoms = jnp.concatenate((dif1, dif2))
            for i in range(single_atoms.shape[0]):
                self._connected_atoms.append([int(single_atoms[i])])


# class Topology:
#
#     def __init__(self, topology: Topology) -> None:
#         self.openmm_topology = topology


class PerveivedTopology:
    # this class implements all possible query actions that depend on the coordinates and elements of the molecular system
    # NOTE: this class is not meant to be used directly, but rather to be inherited by a class that implements the
    def __init__():
        pass

    def get_titratable_atoms(self, coordinates: jnp.array) -> Dict[str, List[int]]:
        pass

    def get_waters(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_protein(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_ligand_atoms(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_center_of_mass(self, coordinates: jnp.array) -> jnp.array:
        pass

    def get_atomic_indices(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_all_unique_elements(self, coordinates: jnp.array) -> Set[int]:
        pass

    def get_connected_protein_graph(self, coordinates: jnp.array) -> List[int]:
        pass

    def get_connected_ligand_graph(self, coordinates: jnp.array) -> List[int]:
        pass
