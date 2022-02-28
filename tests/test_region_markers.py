import pytest

from nanomesh import RegionMarker, RegionMarkerList


class TestRegionMarkerList:

    @pytest.fixture
    def region_markers(self):
        return RegionMarkerList((
            RegionMarker(label=1, point=(1, 2), name='one'),
            RegionMarker(label=1, point=(3, 4), name='one'),
            RegionMarker(label=2, point=(5, 6), name='two'),
            RegionMarker(label=2, point=(7, 8), name='two'),
            RegionMarker(label=3, point=(9, 10), name='three'),
        ))

    def test_region_marker_list(self, region_markers):
        assert isinstance(region_markers, RegionMarkerList)
        assert all(isinstance(m, RegionMarker) for m in region_markers)

    @pytest.mark.parametrize('old,new,expected', (
        (1, 7, (7, 7, 2, 2, 3)),
        (3, 2, (1, 1, 2, 2, 2)),
        (0, 7, (1, 1, 2, 2, 3)),
        ((1, 2), 7, (7, 7, 7, 7, 3)),
        ((0, 2), 7, (1, 1, 7, 7, 3)),
        (lambda x: x == 1, 7, (7, 7, 2, 2, 3)),
        (lambda x: x > 1, 7, (1, 1, 7, 7, 7)),
    ))
    def test_relabel(self, old, new, expected, region_markers):
        new_region_markers = region_markers.relabel(old, new)
        labels = tuple(m.label for m in new_region_markers)
        assert labels == expected

    def test_relabel_name(self, region_markers):
        new_region_markers = region_markers.relabel(1, 7)
        assert new_region_markers.names == {'two', 'three', 'one'}

    @pytest.mark.parametrize('old,expected', (
        (1, (4, 5, 2, 2, 3)),
        (3, (1, 1, 2, 2, 4)),
        (0, (1, 1, 2, 2, 3)),
        ((1, 2), (4, 5, 6, 7, 3)),
        ((0, 2), (1, 1, 4, 5, 3)),
        (lambda x: x == 1, (4, 5, 2, 2, 3)),
        (lambda x: x > 1, (1, 1, 4, 5, 6)),
    ))
    def test_label_sequentially(self, old, expected, region_markers):
        new_region_markers = region_markers.label_sequentially(old)
        labels = tuple(m.label for m in new_region_markers)
        assert labels == expected

    def test_label_sequentially_name(self, region_markers):
        new_region_markers = region_markers.label_sequentially(1,
                                                               fmt_name='f{}')
        assert new_region_markers.names == {'f4', 'f5', 'two', 'three'}

    def test_labels_prop(self, region_markers):
        assert region_markers.labels == {1, 2, 3}

    def test_names_prop(self, region_markers):
        assert region_markers.names == {'one', 'two', 'three'}
