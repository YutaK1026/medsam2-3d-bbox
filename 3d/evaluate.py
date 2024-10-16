from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BoxData:
    z_min: int = 0
    z_max: int = 0
    x_min: int = 0
    x_max: int = 0
    y_min: int = 0
    y_max: int = 0

    def set_value(self, values: List[int]) -> None:
        object.__setattr__(self, "z_min", values[0])
        object.__setattr__(self, "z_max", values[1])
        object.__setattr__(self, "x_min", values[2])
        object.__setattr__(self, "y_min", values[3])
        object.__setattr__(self, "x_max", values[4])
        object.__setattr__(self, "y_max", values[5])

    def get_box_size(self) -> float:
        return (
            (self.z_max - self.z_min)
            * (self.x_max - self.x_min)
            * (self.y_max - self.y_min)
        )


@dataclass
class DimentionOverlap:
    x_overlap: int = 0
    y_overlap: int = 0
    z_overlap: int = 0

    def calcuate_tp(self) -> float:
        if self.x_overlap < 0 or self.y_overlap < 0 or self.z_overlap < 0:
            return 0
        return self.x_overlap * self.y_overlap * self.z_overlap


def evaluate(predict: list[float], test: list[float]) -> int:
    """
    predict
        [z_min, z_max, x_min, y_min, x_max, y_max]
    test
        [z_min, z_max, x_min, y_min, x_max, y_max]
    """
    p_box_data = BoxData()
    t_box_data = BoxData()
    dimention_overlap = DimentionOverlap()
    p_box_data.set_value(predict)
    t_box_data.set_value(test)

    if t_box_data.z_min <= p_box_data.z_max <= t_box_data.z_max:
        z_overlap = p_box_data.z_max - t_box_data.z_min
        p_z_height = p_box_data.z_max - p_box_data.z_min
        t_z_height = t_box_data.z_max - t_box_data.z_min
        dimention_overlap.z_overlap = min(
            z_overlap, p_z_height, t_z_height
        )  # predict, testデータのどちらかが，片方より大きくなる可能性があるため
    elif p_box_data.z_min <= t_box_data.z_max <= p_box_data.z_max:
        z_overlap = t_box_data.z_max - p_box_data.z_min
        p_z_height = p_box_data.z_max - p_box_data.z_min
        t_z_height = t_box_data.z_max - t_box_data.z_min
        dimention_overlap.z_overlap = min(z_overlap, p_z_height, t_z_height)

    if t_box_data.x_min <= p_box_data.x_max <= t_box_data.x_max:
        x_overlap = p_box_data.x_max - t_box_data.x_min
        p_x_height = p_box_data.x_max - p_box_data.x_min
        t_x_height = t_box_data.x_max - t_box_data.x_min
        dimention_overlap.x_overlap = min(x_overlap, p_x_height, t_x_height)
    elif p_box_data.x_min <= t_box_data.x_max <= p_box_data.x_max:
        x_overlap = t_box_data.x_max - p_box_data.x_min
        p_x_height = p_box_data.x_max - p_box_data.x_min
        t_x_height = t_box_data.x_max - t_box_data.x_min
        dimention_overlap.x_overlap = min(x_overlap, p_x_height, t_x_height)

    if t_box_data.y_min <= p_box_data.y_max <= t_box_data.y_max:
        y_overlap = p_box_data.y_max - t_box_data.y_min
        p_y_height = p_box_data.y_max - p_box_data.y_min
        t_y_height = t_box_data.y_max - t_box_data.y_min
        dimention_overlap.y_overlap = min(y_overlap, p_y_height, t_y_height)
    elif p_box_data.y_min <= t_box_data.y_max <= p_box_data.y_max:
        y_overlap = t_box_data.y_max - p_box_data.y_min
        p_y_height = p_box_data.y_max - p_box_data.y_min
        t_y_height = t_box_data.y_max - t_box_data.y_min
        dimention_overlap.y_overlap = min(y_overlap, p_y_height, t_y_height)

    tp = dimention_overlap.calcuate_tp()  # 正しく検出できた
    fn = t_box_data.get_box_size() - tp  # boxを見逃した
    fp = (
        p_box_data.get_box_size() - tp
    )  # bboxの範囲ではないところを，範囲に入れてしまった

    return tp, fn, fp


# テストデータ
predict = [1, 2, 286.0, 167.0, 308.0, 196.0]
test = [1.5, 2, 286.0, 167.0, 308.0, 196.0]

tp, fn, fp = evaluate(predict=predict, test=test)
print(f"tp: {tp}")
print(f"fn: {fn}")
print(f"fp: {fp}")
