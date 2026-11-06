import os.path
from itertools import chain, combinations

import numpy as np
from kikit.common import KiPoint
from kikit.panelize import Panel, expandRect, findBoardBoundingBox, pcbnew, Origin
from kikit.plugin import LayoutPlugin
from kikit.units import mm
from rpack import pack, packing_density, PackingImpossibleError


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def optimal_pack(sizes, max_width=None, max_height=None, allow_rotate=True):
    best_rotate, best_positions, best_density = (), (), -1
    best_rotated_area = 0
    for rotated_sizes_indices in (powerset(range(len(sizes))) if allow_rotate else [()]):
        sizes_with_rotations = [((height, width) if i in rotated_sizes_indices else (width, height))
                                for (i, (width, height)) in enumerate(sizes)]

        try:
            positions = pack(sizes_with_rotations, max_width=max_width, max_height=max_height)
        except PackingImpossibleError:
            continue
        density = packing_density(sizes_with_rotations, positions)
        assert density <= 1.0, "unexpected packing density > 1"
        rotate = [(i in rotated_sizes_indices) for (i, _) in enumerate(sizes_with_rotations)]
        rotated_area = sum(np.array(rotate) * np.array(list(w * h for w, h in sizes)))

        # prefer topologies with less rotational area
        if density > best_density or (density / best_density > (1 - 1e-9) and rotated_area < best_rotated_area):
            best_rotate, best_positions, best_density = rotate, positions, density
            best_rotated_area = rotated_area
            # print('best', best_rotate, best_positions, best_density)

    return best_rotate, best_positions


class Plugin(LayoutPlugin):
    def buildLayout(self, panel: Panel, mainInputFile: str, _sourceArea):
        layout = self.preset["layout"]

        input_yaml: str = layout.get("input", "")
        if not input_yaml:
            raise RuntimeError("Specify the yaml input file like this: --layout '...; input: boards.yaml'")

        import yaml
        with open(input_yaml, 'r') as file:
            yaml_data = yaml.safe_load(file)

        input_boards = yaml_data['boards']
        print(input_boards)

        max_width = yaml_data['max_width'] * mm if 'max_width' in yaml_data else None
        max_height = yaml_data['max_height'] * mm if 'max_height' in yaml_data else None
        allow_rotate = yaml_data.get('allow_rotate', True)
        print(f"max_height: {max_height}, max_width: {max_width}")

        panel.sourcePaths.add(mainInputFile)

        netRenamer = lambda n, orig: self.netPattern.format(n=n, orig=orig)
        refRenamer = lambda n, orig: self.refPattern.format(n=n, orig=orig)

        # TODO use math.gcd
        S = int(layout.get("eps", 1))  # scale extents for better numerical stability, not sure if necessary
        assert S > 0, "eps must be a positive integer"

        sizes = []
        boards = []
        filenames = []
        board_labels = []
        for d in input_boards:
            filename = d['board']
            rotate_deg = float(d.get('rotate', 0))  # pre-rotate TODO
            count = int(d.get('qty', 1))
            assert count > 0, "Count must be > 0"

            margin = float(d.get('margin_mm', 1)) * mm

            if not os.path.isabs(filename):
                filename = os.path.join(os.path.dirname(input_yaml), filename)

            filename = os.path.realpath(filename)

            if not os.path.isfile(filename):
                raise RuntimeError("File '{}' does not exist".format(filename))

            board = pcbnew.LoadBoard(filename)

            tb = board.GetTitleBlock()
            title = tb.GetTitle()
            rev = tb.GetRevision()
            label = f"{title}-{rev}" if title and rev else (title or rev or os.path.basename(filename))
            qty_label = f"{count}x {label}" if count > 1 else label
            if qty_label not in board_labels:
                board_labels.append(qty_label)

            bbox = expandRect(findBoardBoundingBox(board), margin)

            sizes.extend([(
                -(-int(bbox.GetWidth() + self.hspace) // S),
                -(-int(bbox.GetHeight() + self.vspace) // S)
            )] * count)

            boards.extend([board] * count)
            filenames.extend([filename] * count)

        fixed_layout = yaml_data.get('fixed_layout', None)
        if fixed_layout == 'row':
            # Place boards left-to-right in YAML order, no packing
            best_rotates = [False] * len(boards)
            best_positions = []
            x = 0
            for w, h in sizes:
                best_positions.append((x, 0))
                x += w
        elif fixed_layout == 'column':
            # Place boards top-to-bottom in YAML order, no packing
            best_rotates = [False] * len(boards)
            best_positions = []
            y = 0
            for w, h in sizes:
                best_positions.append((0, y))
                y += h
        else:
            pack_max_w = -(-int(max_width) // S) if max_width is not None else None
            pack_max_h = -(-int(max_height) // S) if max_height is not None else None
            best_rotates, best_positions = optimal_pack(sizes, max_width=pack_max_w, max_height=pack_max_h, allow_rotate=allow_rotate)

        # Center boards relative to each other if requested
        if yaml_data.get('center', False) and len(best_positions) > 1:
            if fixed_layout == 'column' or (fixed_layout is None):
                # Center horizontally: find max width, shift each board's x
                max_w = max(sizes[i][0] for i in range(len(sizes)))
                best_positions = [
                    (pos[0] + (max_w - sizes[i][0]) // 2, pos[1])
                    for i, pos in enumerate(best_positions)
                ]
            elif fixed_layout == 'row':
                # Center vertically: find max height, shift each board's y
                max_h = max(sizes[i][1] for i in range(len(sizes)))
                best_positions = [
                    (pos[0], pos[1] + (max_h - sizes[i][1]) // 2)
                    for i, pos in enumerate(best_positions)
                ]

        print(best_rotates, best_positions)

        for i in range(len(boards)):
            panel.appendBoard(
                filename=filenames[i],
                destination=KiPoint(int(best_positions[i][0] * S), int(best_positions[i][1] * S)),
                origin=Origin.TopRight if best_rotates[i] else Origin.TopLeft,
                sourceArea=expandRect(findBoardBoundingBox(boards[i]), 1 * mm),
                netRenamer=netRenamer,
                refRenamer=refRenamer,
                rotationAngle=self.rotation + pcbnew.EDA_ANGLE((90 if best_rotates[i] else 0), pcbnew.DEGREES_T),
                inheritDrc=False,
                bakeText=True,
            )

        # Set panel title block from combined board labels
        panel_tb = panel.board.GetTitleBlock()
        panel_tb.SetTitle(" + ".join(board_labels))
        panel_tb.SetRevision("")
        panel_tb.SetComment(0, f"Panelized from {len(input_boards)} board(s)")
        panel.board.SetTitleBlock(panel_tb)

        print('Done.')

        return panel.substrates


"""

names:
pcb-pack
boardpack
pcbpack

2d rectangle packing problem
https://github.com/Penlect/rectangle-packer
with rotation: https://github.com/Penlect/rectangle-packer/issues/17

TODO optimal packaing
https://stackoverflow.com/questions/1213394/what-algorithm-can-be-used-for-packing-rectangles-of-different-sizes-into-the-sm
https://www.csc.liv.ac.uk/~epa/surveyhtml.html


"""
