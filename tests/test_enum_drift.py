"""Schema ↔ protobuf enum-drift detection.

If someone adds a value to a Pydantic Literal in ``schema.py`` and forgets
to add the corresponding entry in ``proto_bridge.py``, the bridge fails
with a ``KeyError`` at runtime. These tests turn that into an import-time
test failure instead.
"""

from __future__ import annotations

from typing import get_args

import pytest

from eln_structurer import proto_bridge, schema


# Each entry: a Pydantic Literal alias from schema.py and the proto map(s)
# in proto_bridge.py whose union of keys must contain every value.
_ENUM_PROTO_PAIRS = [
    (
        "IdentifierType",
        schema.IdentifierType,
        [proto_bridge._IDENTIFIER_TYPE_PROTO],
    ),
    (
        "ReactionRole",
        schema.ReactionRole,
        [proto_bridge._ROLE_PROTO],
    ),
    (
        "TempControlType",
        schema.TempControlType,
        [proto_bridge._TEMP_CONTROL_PROTO],
    ),
    (
        "StirringType",
        schema.StirringType,
        [proto_bridge._STIRRING_PROTO],
    ),
    (
        "WorkupType",
        schema.WorkupType,
        [proto_bridge._WORKUP_TYPE_PROTO],
    ),
    (
        "ProductMeasurementType",
        schema.ProductMeasurementType,
        [proto_bridge._MEASUREMENT_TYPE_PROTO],
    ),
    (
        # AmountUnit is special: it's covered by the union of mass/moles/
        # volume maps plus the manual {equiv, mass_pct, mol_pct, vol_pct}
        # set that _apply_amount handles via the unmeasured branch.
        "AmountUnit",
        schema.AmountUnit,
        [
            proto_bridge._MASS_UNIT_PROTO,
            proto_bridge._MOLES_UNIT_PROTO,
            proto_bridge._VOLUME_UNIT_PROTO,
            {"equiv": None, "mass_pct": None, "mol_pct": None, "vol_pct": None},
        ],
    ),
]


@pytest.mark.parametrize("enum_name,enum_alias,proto_maps", _ENUM_PROTO_PAIRS)
def test_pydantic_enum_covered_by_proto_map(enum_name, enum_alias, proto_maps) -> None:
    expected = set(get_args(enum_alias))
    covered: set[str] = set()
    for m in proto_maps:
        covered |= set(m.keys())
    missing = expected - covered
    assert not missing, (
        f"{enum_name} values not handled by proto_bridge: {sorted(missing)}"
    )
