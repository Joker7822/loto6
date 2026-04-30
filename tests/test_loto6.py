import pandas as pd
import pytest

from loto6 import Loto6Predictor, classify_loto6, normalize_numbers


def sample_df(rows=360):
    records = []
    for i in range(1, rows + 1):
        nums = sorted({((i + j * 7) % 43) + 1 for j in range(6)})
        x = 1
        while len(nums) < 6:
            if x not in nums:
                nums.append(x)
            x += 1
        nums = sorted(nums[:6])
        bonus = ((i * 5) % 43) + 1
        if bonus in nums:
            bonus = (bonus % 43) + 1
        records.append({"draw_no": i, "date": f"2020-01-{(i % 28) + 1:02d}", **{f"n{j + 1}": nums[j] for j in range(6)}, "bonus": bonus})
    return pd.DataFrame(records)


def test_normalize_numbers_validation():
    assert normalize_numbers([6, 1, 43, 2, 3, 4]) == (1, 2, 3, 4, 6, 43)
    with pytest.raises(ValueError):
        normalize_numbers([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        normalize_numbers([1, 2, 3, 4, 5, 5])
    with pytest.raises(ValueError):
        normalize_numbers([1, 2, 3, 4, 5, 44])


def test_classify_loto6_grades():
    main = [1, 2, 3, 4, 5, 6]
    assert classify_loto6([1, 2, 3, 4, 5, 6], main, 7).grade == "1等"
    assert classify_loto6([1, 2, 3, 4, 5, 7], main, 7).grade == "2等"
    assert classify_loto6([1, 2, 3, 4, 5, 8], main, 7).grade == "3等"
    assert classify_loto6([1, 2, 3, 4, 9, 10], main, 7).grade == "4等"
    assert classify_loto6([1, 2, 3, 9, 10, 11], main, 7).grade == "5等"
    assert classify_loto6([1, 2, 9, 10, 11, 12], main, 7).grade == "はずれ"


def test_predictor_returns_valid_combinations():
    preds = Loto6Predictor(seed=1).fit(sample_df()).predict(n=5, candidate_count=500)
    assert len(preds) == 5
    assert [p.rank for p in preds] == [1, 2, 3, 4, 5]
    for pred in preds:
        assert len(pred.numbers) == 6
        assert len(set(pred.numbers)) == 6
        assert min(pred.numbers) >= 1
        assert max(pred.numbers) <= 43
