# Standard Library Imports
import io
import random
from collections import Counter
from enum import Enum
from itertools import product
from tkinter import messagebox

# Third-Party Imports
import cairosvg
import graphviz
import numpy as np
from PIL import Image

# PM4Py Imports
import pm4py
from pm4py.objects.dfg.utils import dfg_utils
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.objects.heuristics_net.obj import HeuristicsNet
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.statistics.attributes.log import get as log_attributes
from pm4py.util import constants, exec_utils, xes_constants
from pm4py.util import xes_constants as xes
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.heuristics_net import converter as hn_converter  # new

# Type Checking (Conditional Import)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from GUI import GUI as GUI


class DPHM:
    class Parameters(Enum):
        # Source: Based on and abbreviated from PM4PY
        ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY

    def __init__(self, gui):
        self.event_log = None
        self.parameters = {}
        self.activity_key = exec_utils.get_param_value(self.Parameters.ACTIVITY_KEY, self.parameters,
                                                       xes.DEFAULT_NAME_KEY)  # Source: PM4PY
        self.activities = None
        self.trace_list = None
        self.df_relations = None
        self.matrix = None
        self.noised_matrix = None
        self.starting_activities = None
        self.ending_activities = None

        self.tree = None
        self.net = None
        self.im = None
        self.fm = None

        self.gamma: float = 0.01
        self.e_0: float = 0.01
        self.max_sampling_tries: int = int(max(1 / self.gamma * np.log(2 / self.e_0), 1 / (np.e * self.gamma)))

        self.GUI: GUI = gui

    def add_event_log(self, log):
        try:
            self.event_log = xes_importer.apply(log)
            self.activities = None
            self.trace_list = None
            self.df_relations = None
            self.matrix = None
            self.noised_matrix = None
            self.starting_activities = None
            self.ending_activities = None
            self.extract_activities()

        except Exception as e:
            messagebox.showerror("Error", f"Event log could not be loaded: {e}")

    def extract_activities(self):
        self.activities = list(
            log_attributes.get_attribute_values(
                self.event_log,
                self.activity_key,
                parameters=self.parameters
            ).keys())
        self.get_trace_list()

    def get_trace_list(self):
        trace_list = list()
        act_key = exec_utils.get_param_value(self.Parameters.ACTIVITY_KEY.value, parameters={},
                                             default=xes_constants.DEFAULT_NAME_KEY)
        for trace in self.event_log:
            tmp_list = list()
            for i in range(len(trace) - 1):
                if i == 0:  # first activity
                    tmp_list.append(('0xb2e-start-0x31c', trace[i][act_key]))  # pre-fix a synthetic start
                tmp_list.append((trace[i][act_key], trace[i + 1][act_key]))  # in-between activity

                if i == len(trace) - 2:  # last activity
                    tmp_list.append((trace[i + 1][act_key], '0x31c-end-0x1021'))  # post-fix a synthetic end

            if len(trace) == 1:  # consider traces of length 1
                tmp_list.append(('0xb2e-start-0x31c', trace[0][act_key]))
                tmp_list.append((trace[0][act_key], '0x31c-end-0x1021'))

            tmp_list = list(set(tmp_list))  # upper-bind sensitivity to 1
            trace_list.append(tmp_list)

        self.trace_list = trace_list
        self.create_matrix()

    def create_matrix(self):
        activities = sorted(self.activities, key=str.lower)
        permutations = list(product(self.activities, repeat=2))
        for act in activities:
            permutations.append(tuple(('0xb2e-start-0x31c', act)))
            permutations.append(tuple((act, '0x31c-end-0x1021')))
        temp_matrix = {perm: 0 for perm in permutations}

        matrix = Counter(temp_matrix)

        self.matrix = matrix
        self.fill_matrix()

    def fill_matrix(self):
        for trace in self.trace_list:
            for pair in trace:
                self.matrix[pair] += 1

        self.rejection_sampling()

    def noise_matrix(self):
        if self.event_log is None:
            return

        # Preparation
        noised_matrix = {}
        starting_activities = {a: 0 for a in self.activities}
        ending_activities = {a: 0 for a in self.activities}

        # Noise everything
        for key, value in self.matrix.items():
            noised_matrix[key] = int(self.add_laplace_noise(value, 1, self.GUI.epsilon.get()*0.65))

        # Extract noised starting and ending activities
        for key, value in noised_matrix.items():
            if key[0] == '0xb2e-start-0x31c':
                starting_activities[key[1]] += value
            if key[1] == '0x31c-end-0x1021':
                ending_activities[key[0]] += value

        # Create subsets with Report Noisy Max
        s: int = np.random.randint(1, len(starting_activities))
        starting_activities = self.report_noisy_max(starting_activities, s, self.GUI.epsilon.get()*0.25)
        e: int = np.random.randint(1, len(ending_activities))
        ending_activities = self.report_noisy_max(ending_activities, e, self.GUI.epsilon.get()*0.25)

        # Remove non-positive Starting/Ending counts:
        starting_activities = {k: v for k, v in starting_activities.items() if v > 0}
        ending_activities = {k: v for k, v in ending_activities.items() if v > 0}

        # Remove synthetic start and end activities from matrix and convert it to counter
        filtered_dict = {k: v for k, v in noised_matrix.items() if
                         '0xb2e-start-0x31c' not in k and '0x31c-end-0x1021' not in k}

        # Create subset of matrix of all behavior
        lower, upper = self.calculate_bounds(filtered_dict)
        b: int = np.random.randint(lower, upper)
        filtered_dict = self.report_noisy_max(filtered_dict, b, self.GUI.epsilon.get()*0.25)
        noised_matrix = Counter(filtered_dict)

        self.noised_matrix = noised_matrix
        self.starting_activities = starting_activities
        self.ending_activities = ending_activities

        # self.rejection_sampling()

    def add_laplace_noise(self, original_value: float, sensitivity: float, epsilon: float) -> float:
        scale = sensitivity / epsilon
        noise = np.random.laplace(0., scale)
        noised_val = original_value + noise

        return noised_val

    def calculate_bounds(self, matrix: dict):
        # Count the number of activity pairs with a frequency > 0
        count_above_zero = sum(1 for value in matrix.values() if value > 0)

        # Calculate preliminary lower and upper bounds
        lower_bound = count_above_zero - 15
        upper_bound = count_above_zero + 15

        # Round to the nearest multiple of 5
        lower_bound = max(count_above_zero, 5 * round(lower_bound / 5))
        upper_bound = min(count_above_zero ** 2, 5 * round(upper_bound / 5))

        # Set boundaries for lower and upper bound
        if lower_bound < len(self.activities):
            lower_bound = len(self.activities)

        if upper_bound >= len(self.activities)**2:
            upper_bound = len(self.activities)**2-1

        return lower_bound, upper_bound

    def report_noisy_max(self, list: dict, n: int, epsilon: float ):
        if epsilon is None:
            epsilon = self.GUI.epsilon.get()

        if not list or n <= 0:
            return {}

        # Select the top-n highest noisy scores
        top_n_items = {key: list[key] for key in sorted(list, key=list.get, reverse=True)[:n]}

        return top_n_items

    def rejection_sampling(self, renoise: bool=True):

        if self.event_log is None:
            return False

        for i in range(0, self.max_sampling_tries):
            # probability to stop and return nothing
            coin_flip = random.random()
            if coin_flip <= self.gamma:
                return False

            if renoise:
                self.noise_matrix()

            noised_heu_net = HeuristicsNet(
                frequency_dfg=self.noised_matrix,  # safe, because epsilon-DP-noised
                activities=self.activities,  # safe, because we do not intend to change the process domain
                activities_occurrences=None,  # safe, because based on epsilon-DP-noised activity counts
                start_activities=self.starting_activities,  # safe, because epsilon-DP-noised
                end_activities=self.ending_activities,  # safe, because epsilon-DP-noised
                dfg_window_2=None,  # safe, because None
                freq_triples=None,  # safe, because None
                performance_dfg=None  # safe, because None
            )

            self.noised_heu_net = pm4py.algo.discovery.heuristics.variants.classic.calculate(
                heu_net=noised_heu_net,  # safe, because epsilon-DP-noised
                parameters={},  # safe, because {}
                dependency_thresh=self.GUI.dependency.get(),
                and_measure_thresh=self.GUI.AND.get(),
                min_act_count=self.GUI.min_act.get(),
                min_dfg_occurrences=self.GUI.min_dfg.get(),
                dfg_pre_cleaning_noise_thresh=self.GUI.pre_noise.get(),
                loops_length_two_thresh=self.GUI.loop2.get()
            )

            try:
                # Convert HeuristicsNet to Petri net
                n, im, fm = pm4py.algo.discovery.heuristics.variants.classic.hn_conv_alg.apply(
                    noised_heu_net, parameters=self.parameters)

                # Convert Petri net to process tree
                t = pm4py.convert_to_process_tree(n, im, fm)
                self.tree = t
                self.net = n
                self.im = im
                self.fm = fm

            except ValueError:
                pass

            if self.check_rejection():
                return

    def check_rejection(self) -> bool:
            # Caching values for rejection sampling
            rej_sam_attr: str = self.GUI.rejection_sampling_attr.get()
            thresh_value: float = self.GUI.rejection_threshold.get()

            if rej_sam_attr == "Fitness":
                try:
                    if self.tree is not None:
                        fitness_tb = pm4py.fitness_token_based_replay(self.event_log, self.net, self.im, self.fm)
                        if (self.add_laplace_noise(fitness_tb.get('log_fitness'),
                                                   1, self.GUI.epsilon.get() * 0.1) >= thresh_value):
                            self.render()
                            return True

                except ValueError:
                    pass

            elif rej_sam_attr == "Precision":
                try:
                    if self.tree is not None:
                        precision_tb = pm4py.precision_token_based_replay(self.event_log, self.net, self.im, self.fm)
                        if (self.add_laplace_noise(precision_tb,
                                                   1, self.GUI.epsilon.get() * 0.1) >= thresh_value):
                            self.render()
                            return True

                except ValueError:
                    pass

            elif rej_sam_attr == "Simplicity":
                try:
                    if self.tree is not None:
                        simplicity = simplicity_evaluator.apply(self.net)
                        if (self.add_laplace_noise(simplicity,
                                                   1, self.GUI.epsilon.get() * 0.1) >= thresh_value):
                            self.render()
                            return True

                except ValueError:
                    pass

            elif rej_sam_attr == "Generalization":
                try:
                    if self.tree is not None:
                        generalization = generalization_evaluator.apply(self.event_log, self.net, self.im, self.fm)
                        if (self.add_laplace_noise(generalization,
                                                   1, self.GUI.epsilon.get() * 0.1) >= thresh_value):
                            self.render()
                            return True

                except ValueError:
                    pass

            elif rej_sam_attr == "F1-Score":
                try:
                    if self.tree is not None:
                        fitness_tb = pm4py.fitness_token_based_replay(self.event_log, self.net, self.im, self.fm)
                        precision_tb = pm4py.precision_token_based_replay(self.event_log, self.net, self.im, self.fm)
                        f1 = (fitness_tb.get('log_fitness') + precision_tb) / 2
                        if (self.add_laplace_noise(f1,
                                                   1, self.GUI.epsilon.get() * 0.1) >= thresh_value):
                            self.render()
                            return True

                except ValueError:
                    pass

            return False

    def render(self):
        if self.tree is not None:

            # 'Dependency Graph':
            dot = graphviz.Digraph(format="png")
            for act1 in self.noised_heu_net.dependency_matrix:
                for act2 in self.noised_heu_net.dependency_matrix[act1]:
                    weight = self.noised_heu_net.dependency_matrix[act1][act2]
                    if weight > self.GUI.dependency.get():
                        dot.edge(act1, act2, label=str(round(weight, 2)))
            png_data = dot.pipe(format="png")
            img = Image.open(io.BytesIO(png_data))
            self.GUI.apply_image(img, 1)

            # 'Petri Net':
            viz = pn_visualizer.apply(self.net, self.im, self.fm, parameters={
                pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"})
            svg_data = pm4py.visualization.petri_net.visualizer.serialize(viz)
            png_data = cairosvg.svg2png(bytestring=svg_data)
            img = Image.open(io.BytesIO(png_data))
            self.GUI.apply_image(img, 2)

            # 'BPMN':
            bpmn_graph = pm4py.convert_to_bpmn(self.tree)
            viz_bpmn = bpmn_visualizer.apply(bpmn_graph)
            svg_data = bpmn_visualizer.serialize(viz_bpmn)
            if svg_data[:4] == b'\x89PNG':
                img = Image.open(io.BytesIO(svg_data))
            else:
                png_data = cairosvg.svg2png(bytestring=svg_data.encode("utf-8"))
                img = Image.open(io.BytesIO(png_data))
            self.GUI.apply_image(img, 3)

            # 'Process Tree':
            viz = pt_visualizer.apply(self.tree, parameters={
                pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "svg"})
            svg_data = pt_visualizer.serialize(viz)
            png_data = cairosvg.svg2png(bytestring=svg_data)
            img = Image.open(io.BytesIO(png_data))
            self.GUI.apply_image(img, 4)
            print("Done executing visualizations.")

        else:
            print("No Tree provided.")


if __name__ == '__main__':
    from GUI import GUI
    app = GUI()
    app.root.mainloop()
