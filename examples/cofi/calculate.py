# command line script
from promis import ProMis, DeltaStaRMap
from promis.geo import DeltaGrid, PolarLocation
from pathlib import Path

general_logic = """
    unclassified_side_correct(X) :-
        follows(X, unclassified),
        on_right_side(X, unclassified);
        \+follows(X, unclassified),
        \+on_right_side(X, unclassified).

    unclassified_correct(X) :-
        over(X, unclassified),
        unclassified_side_correct(X),
        state_speed(X, S),
        maxspeed(X, unclassified, MS),
        MS >= S.

    tertiary_side_correct(X) :-
        follows(X, tertiary),
        on_right_side(X, tertiary);
        \+follows(X, tertiary),
        \+on_right_side(X, tertiary).

    tertiary_correct(X) :-
        over(X, tertiary),
        tertiary_side_correct(X),
        state_speed(X, S),
        maxspeed(X, tertiary, MS),
        MS >= S.
        

    % Definition of a valid mission
    landscape(X) :-
        tertiary_correct(X);
        unclassified_correct(X).
"""
origin = PolarLocation(latitude=50.782031183109694, longitude=6.071167919731977)
width, height = 60, 60

output_folder = Path('ground-cofi-exports/inD-crossing-08')

support = DeltaGrid(origin, (40, 40), width, height, speed_res=3, speed_bounds=(35, 55), bearing_res=13)
dsm = DeltaStaRMap.load(output_folder / "dsm.pkl")
promis = ProMis(dsm)
promis.solve(support, general_logic, print_first=True, show_progress=True)
support.save(output_folder / "landscape.pkl")
