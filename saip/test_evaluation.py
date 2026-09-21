import numpy as np
from saip.evaluation import evaluate, iou_matrix, token_f1


def test_seconds_iou_and_disjoint_intervals():
    assert np.allclose(iou_matrix([[0,10]],[[5,15],[20,30]]),[[1/3,0]])


def test_duplicates_do_not_inflate_event_count_and_missing_videos_count():
    ref = {'a':dict(duration=10,timestamps=[[1,4],[1,4]],sentences=['person runs','a person runs']),
           'b':dict(duration=10,timestamps=[[1,4]],sentences=['walk'])}
    pred = {'a':dict(timestamps=[[1,4],[1,4]],sentences=['person runs','person runs'])}
    result = evaluate(pred,ref,['a','b'])['summary']
    assert result['unique_reference_intervals'] == 2
    assert result['recall@0.5'] == .5
    assert result['precision@0.5'] == .5
    assert token_f1('person runs','person runs') == 1


def test_maximum_matching_not_greedy():
    ref = {'a':dict(duration=10,timestamps=[[0,4],[3,7]],sentences=['a','b'])}
    pred = {'a':dict(timestamps=[[0,7],[0,3]],sentences=['a','b'])}
    assert evaluate(pred,ref,['a'])['summary']['recall@0.3'] == 1
