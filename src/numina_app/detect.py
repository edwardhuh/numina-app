"""Helper functions that assist with the detection and classification of objects"""
import numpy as np
import pandas as pd


def get_zones(behavior_zones: pd.DataFrame, bottom_center: pd.Series) -> pd.Series:
    """Return a categorical variable for each entry's associated behavior zone,
    Args:
        behavior_zones
        bottom_center:  `bottom_center` column from original data

    Returns:
        A Series indicating which behavior zone a bottom_center belongs to. NaN otherwise.

    Example:
        >>> import pandas as pd
        >>> import schema
        >>> behavior_zones = pd.read_csv("../../data/behavior_zones.csv")
        >>> data = pd.read_csv("../../data/data.csv")
        >>> data = schema.format_data(data)
        >>> get_zones(behavior_zones=behavior_zones, bottom_center=data.bottom_center)
            0             NaN
            1             NaN
            2             NaN
            3             NaN
            4        sidewalk
                    ...
            29899         NaN
            29900         NaN
            29901         NaN
            29902    sidewalk
            29903    sidewalk
            Name: where_am_i, Length: 29904, dtype: object

    """
    bottom_x = bottom_center.map(lambda x: x[0])
    bottom_y = bottom_center.map(lambda x: x[1])
    within_dict = {}

    # Create dummy variable for each behavior zone
    for _, row in behavior_zones.iterrows():
        val = np.logical_and(
            bottom_x.between(row["xmin"], row["xmax"]),
            bottom_y.between(row["ymin"], row["ymax"]),
        )
        within_dict[row["behavior_zone"]] = val

    # Collapse dummy variable
    data = pd.DataFrame.from_dict(within_dict)
    within_cols = data.columns
    data.loc[:, "w_any"] = data.sum(axis=1)
    assert data.w_any.max() == 1

    # When collapsing, use `w_any` column as mask to ensure that NaNs are introduced during left merge.
    data = data.merge(
        pd.Series(
            data[within_cols].idxmax(axis=1)[data.w_any.astype(bool)], name="where_am_i"
        ),
        how="left",
        left_index=True,
        right_index=True,
    )
    return data.where_am_i
