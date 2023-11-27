from pathlib import Path
from typing import List

from matplotlib.pyplot import figure, grid, show, plot, legend

from dataset.physionet2022challenge.extended.labels import murmur_classes, outcome_classes
from dataset.physionet2022challenge.extended.patient import Patient, load_patients
from obsoletestuff.demographics import exp1

if __name__ == '__main__':
    dataset_path: Path = Path(
        '/home/vasileios/workspace/Datasets/physionet2022/physionet.org/files/circor-heart-sound/1.0.3/training_data/')
    patients: List[Patient] = load_patients(dataset_path)

    h_murmur_ages = {}
    for murmur in murmur_classes:
        h_murmur_ages[murmur] = {}
        for age in exp1._ages:
            h_murmur_ages[murmur][age] = len([x for x in patients if x.age == age and x.murmur == murmur])

    h_murmur_sex = {}
    for murmur in murmur_classes:
        h_murmur_sex[murmur] = {}
        for sex in exp1._sexes:
            h_murmur_sex[murmur][sex] = len([x for x in patients if x.sex == sex and x.murmur == murmur])

    h_murmur_pregnancy = {}
    for murmur in murmur_classes:
        h_murmur_pregnancy[murmur] = {}
        for pregnancy in [True, False]:
            h_murmur_pregnancy[murmur][pregnancy] = len([x for x in patients if x.pregnancy == pregnancy and x.murmur == murmur])

    #

    h_outcome_ages = {}
    for outcome in outcome_classes:
        h_outcome_ages[outcome] = {}
        for age in exp1._ages:
            h_outcome_ages[outcome][age] = len([x for x in patients if x.age == age and x.outcome == outcome])

    h_outcome_sex = {}
    for outcome in outcome_classes:
        h_outcome_sex[outcome] = {}
        for sex in exp1._sexes:
            h_outcome_sex[outcome][sex] = len([x for x in patients if x.sex == sex and x.outcome == outcome])

    h_outcome_pregnancy = {}
    for outcome in outcome_classes:
        h_outcome_pregnancy[outcome] = {}
        for pregnancy in [True, False]:
            h_outcome_pregnancy[outcome][pregnancy] = len([x for x in patients if x.pregnancy == pregnancy and x.outcome == outcome])

    #

    figure()
    for murmur in murmur_classes:
        p = [x for x in patients if x.murmur == murmur]
        plot([x.weight for x in p],[x.height for x in p], '.')
    grid()
    legend(murmur_classes)
    show()

    figure()
    for outcome in outcome_classes:
        p = [x for x in patients if x.outcome == outcome]
        plot([x.weight for x in p],[x.height for x in p], '.')
    grid()
    legend(outcome_classes)
    show()

    print('done')
