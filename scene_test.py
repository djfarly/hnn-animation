import random

import numpy as np

from manim import *
from manim_neural_network.neural_network import NeuralNetworkMobject


class MNISTInputLayerMobject(NeuralNetworkMobject):
    """First layer: 4 neurons with ellipsis in place of the middle (3rd) neuron."""

    def get_layer(self, size, index=-1):
        if index != 0:
            return super().get_layer(size, index)
        # Input layer: 4 neurons, ellipsis in the middle
        layer = VGroup()
        neurons = VGroup(*[
            Circle(
                radius=self.neuron_radius,
                stroke_color=self.get_nn_fill_color(index),
                stroke_width=self.neuron_stroke_width,
                fill_color=BLACK,
                fill_opacity=self.neuron_fill_opacity,
            )
            for _ in range(4)
        ])
        for neuron in neurons:
            neuron.z_index = 1
        ellipsis = Text("…", font_size=20)
        # Arrange: top 2 neurons, ellipsis, bottom 2 neurons
        top = VGroup(neurons[0], neurons[1]).arrange(DOWN, buff=self.neuron_to_neuron_buff)
        bottom = VGroup(neurons[2], neurons[3]).arrange(DOWN, buff=self.neuron_to_neuron_buff)
        column = VGroup(top, ellipsis, bottom).arrange(DOWN, buff=self.neuron_to_neuron_buff)
        layer.neurons = neurons
        layer.add(column)
        return layer


class NNDemo(Scene):
    def construct(self):
        nn = MNISTInputLayerMobject([12, 10, 10, 10])
        nn.layer_to_layer_buff = 1.4
        nn.scale(0.68)
        nn.shift(RIGHT * 1.5)

        # Apply initial styling before animations
        for edge_group in nn.edge_groups:
            for edge in edge_group:
                edge.set_stroke(color=GREY_C)

        # Fraction of edges to keep visible per group (input→H1, H1→H2, H2→output)
        edge_visibility = [0.5, 0.4, 0.4, 0.4]
        for edge_group, frac in zip(nn.edge_groups, edge_visibility):
            edges = list(edge_group)
            n_hide = int(len(edges) * (1 - frac))
            if n_hide > 0:
                for edge in random.sample(edges, n_hide):
                    edge.set_opacity(0)

        input_layer = nn.layers[0]
        input_bits = [1, 0, 1, 0]
        input_on_color = YELLOW
        input_off_color = GREY_D
        for neuron, bit in zip(input_layer.neurons, input_bits):
            if bit:
                neuron.set_fill(color=input_on_color, opacity=0.9)
                neuron.set_stroke(color=input_on_color, width=2.4)
            else:
                neuron.set_fill(color=input_off_color, opacity=0.15)
                neuron.set_stroke(color=input_off_color, width=1.6)

        hidden_layer_1 = nn.layers[1]
        hidden_layer_2 = nn.layers[2]
        output_layer = nn.layers[-1]

        h1_color = TEAL_B
        h2_color = ORANGE

        for neuron in hidden_layer_1.neurons:
            neuron.set_fill(color=h1_color, opacity=0.08)
            neuron.set_stroke(color=h1_color, width=1.6, opacity=0.6)
        for neuron in hidden_layer_2.neurons:
            neuron.set_fill(color=h2_color, opacity=0.08)
            neuron.set_stroke(color=h2_color, width=1.6, opacity=0.6)

        for neuron in output_layer.neurons:
            neuron.set_fill(color=GREY_D, opacity=0.15)
            neuron.set_stroke(color=GREY_D, width=1.8)

        image = ImageMobject("mnist-binary-14x14-class-5.png")
        image.pixelated = True
        image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        image.set(height=2.15)

        grid = VGroup()
        grid.set_z_index(image.z_index + 1)
        grid_size = 14
        cell_width = image.width / grid_size
        cell_height = image.height / grid_size
        half_w = image.width / 2
        half_h = image.height / 2
        for i in range(1, grid_size):
            x = -half_w + i * cell_width
            vertical = Line(
                start=np.array([x, -half_h, 0]),
                end=np.array([x, half_h, 0]),
                stroke_color=GREY_B,
                stroke_width=0.8,
                stroke_opacity=0.55,
            )
            grid.add(vertical)
        for j in range(1, grid_size):
            y = -half_h + j * cell_height
            horizontal = Line(
                start=np.array([-half_w, y, 0]),
                end=np.array([half_w, y, 0]),
                stroke_color=GREY_B,
                stroke_width=0.8,
                stroke_opacity=0.55,
            )
            grid.add(horizontal)
        border = Rectangle(
            width=image.width,
            height=image.height,
            stroke_color=GREY_B,
            stroke_width=1.0,
            stroke_opacity=0.8,
        )
        grid.add(border)
        grid.move_to(image)

        size_hint = Text("196 Pixels", font_size=20, color=GREY_A)

        layer_labels = ["Input", "H1", "H2", "Output"]
        label_mobs = VGroup(
            *[
                Text(text, font_size=20).next_to(layer, DOWN, buff=0.35)
                for layer, text in zip(nn.layers, layer_labels)
            ]
        )

        output_digit_labels = VGroup(
            *[
                Text(str(i), font_size=20).next_to(neuron, RIGHT, buff=0.2)
                for i, neuron in enumerate(nn.layers[-1].neurons)
            ]
        )

        prediction_text = VGroup(
            Text("Prediction: 5", font_size=28, color=GREEN_B, weight="BOLD"),
            Text("Confidence: ~85%", font_size=22, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        prediction_text.next_to(nn.layers[-1], RIGHT, buff=1.1)

        image.next_to(nn.layers[0], LEFT, buff=1.1)
        grid.move_to(image)
        size_hint.next_to(image, DOWN, buff=0.2)

        layout_group = Group(image, grid, size_hint, nn, label_mobs, output_digit_labels, prediction_text)
        layout_group.center()

        final_image_pos = image.get_center()
        final_grid_pos = grid.get_center()
        image.move_to(ORIGIN)
        grid.move_to(image)

        self.play(FadeIn(image), FadeIn(grid))
        self.wait(0.3)
        self.play(image.animate.move_to(final_image_pos), grid.animate.move_to(final_grid_pos))
        grid.move_to(image)
        size_hint.next_to(image, DOWN, buff=0.2)
        self.play(Write(size_hint))
        self.play(Create(nn))
        self.play(Write(label_mobs), Write(output_digit_labels))
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    (
                        hidden_layer_1.neurons[i]
                        .animate.set_fill(color=h1_color, opacity=0.9)
                        .set_stroke(color=h1_color, width=2.6, opacity=1.0)
                    )
                    if i in {0, 2, 4, 7}
                    else hidden_layer_1.neurons[i]
                    .animate.set_fill(color=h1_color, opacity=0.05)
                    .set_stroke(color=h1_color, width=1.2, opacity=0.25)
                    for i in range(len(hidden_layer_1.neurons))
                ],
                lag_ratio=0.1,
                run_time=2.0,
            )
        )
        self.play(
            LaggedStart(
                *[
                    (
                        hidden_layer_2.neurons[i]
                        .animate.set_fill(color=h2_color, opacity=0.9)
                        .set_stroke(color=h2_color, width=2.6, opacity=1.0)
                    )
                    if i in {1, 3, 5, 8}
                    else hidden_layer_2.neurons[i]
                    .animate.set_fill(color=h2_color, opacity=0.05)
                    .set_stroke(color=h2_color, width=1.2, opacity=0.25)
                    for i in range(len(hidden_layer_2.neurons))
                ],
                lag_ratio=0.1,
                run_time=2.0,
            )
        )

        self.play(
            LaggedStart(
                *[
                    output_layer.neurons[i]
                    .animate.set_fill(color=BLUE_E, opacity=0.3)
                    .set_stroke(color=BLUE_E, width=2.0, opacity=0.8)
                    for i in range(len(output_layer.neurons))
                ],
                lag_ratio=0.08,
                run_time=1.6,
            )
        )

        class_five_neuron = output_layer.neurons[5]
        class_three_neuron = output_layer.neurons[3]
        subdued_indices = [i for i in range(len(output_layer.neurons)) if i not in (5, 3)]

        self.play(
            class_five_neuron.animate.set_fill(color=GREEN_B, opacity=1.0).set_stroke(color=GREEN_A, width=3.0),
            output_digit_labels[5].animate.set_color(GREEN_B),
            class_three_neuron.animate.set_fill(color=YELLOW_E, opacity=0.5).set_stroke(color=YELLOW_E, width=2.5),
            output_digit_labels[3].animate.set_color(YELLOW_E),
            *[
                output_layer.neurons[i]
                .animate.set_fill(color=GREY_B, opacity=0.25)
                .set_stroke(color=GREY_B, width=1.8)
                for i in subdued_indices
            ],
            run_time=1.6,
        )

        self.play(Indicate(class_five_neuron, color=GREEN_B, scale_factor=1.1), run_time=1.2)
        self.play(*[Write(line) for line in prediction_text])
        self.wait(2)
