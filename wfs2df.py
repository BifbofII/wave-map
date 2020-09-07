import os.path
import dateutil.parser
import pandas as pd
import xml.etree.ElementTree as et

def parse_wfs(in_str):
    """
    Parse XML data in GML format of a WFS server to a pandas dataframe

    The data is parsed and returned as a pandas dataframe.
    Additionally some of the metadata is returned as a dict.

    Currently only parsing of time value pair data is supported.

    :param in_str: either the path to an xml file or a string containing xml data
    :returns: pandas dataframe containing the data and a dict containing additional metadata
    :raises ValueError: when invalid xml data is given
    """
    if os.path.isfile(in_str):
        tree = et.parse(in_str)
        xml_root = tree.getroot()
    else:
        xml_root = et.fromstring(in_str)

    if xml_root.tag != '{http://www.opengis.net/wfs/2.0}FeatureCollection':
        raise ValueError('Invalid GML data')

    metadata = xml_root.attrib
    del metadata['{http://www.w3.org/2001/XMLSchema-instance}schemaLocation']

    data = []
    for member in xml_root.iterfind('{http://www.opengis.net/wfs/2.0}member'):
        data.append(_parse_member(member))

    return pd.concat(data), metadata


def _parse_member(member):
    """
    Parse a WFS member element

    :param member: a member xml element
    :returns: a pandas dataframe with the data of the member
    """
    data = []
    for ts in member.iterfind('{http://inspire.ec.europa.eu/schemas/omso/3.0}PointTimeSeriesObservation'):
        data.append(_parse_ts(ts))

    return pd.concat(data)

        
def _parse_ts(ts):
    """
    Parse a time series of the GML time value pair format

    :param ts: the time series xml element
    :returns: a pandas dataframe with the time series data
    """
    result = ts.find('{http://www.opengis.net/om/2.0}result')
    if not result:
        ValueError('Time series contains no result data')

    sers = []
    for mt in result.iterfind('{http://www.opengis.net/waterml/2.0}MeasurementTimeseries'):
        points = [_parse_tvp(p.find('{http://www.opengis.net/waterml/2.0}MeasurementTVP')) for p in mt]
        data = {i: [p[i] for p in points] for i in points[0]}
        sers.append(pd.Series(data=data['value'], index=data['time'], name=mt.attrib['{http://www.opengis.net/gml/3.2}id']))

    return pd.concat(sers, axis=1)


def _parse_time_period(tp):
    """
    Parse a GML time period

    :param tp: a GML time period xml element
    :returns: a dict with the entries begin and end as datetimes
    """
    time = dict()
    time['begin'] = dateutil.parser.isoparse(tp.find('{http://www.opengis.net/gml/3.2}beginPosition').text)
    time['begin'] = dateutil.parser.isoparse(tp.find('{http://www.opengis.net/gml/3.2}endPosition').text)
    return time

def _parse_time_instant(ti):
    """
    Parse a GML time instant

    :param ti: a GML time instant xml element
    :returns: a datatime
    """
    return dateutil.parser.isoparse(ti.find('{http://www.opengis.net/gml/3.2}timePosition').text)

def _parse_tvp(tvp):
    """
    Parse a TVP point

    :param p: a TVP xml element
    :returns: a dict with time and value entries
    """
    result = dict()
    result['time'] = dateutil.parser.isoparse(tvp.find('{http://www.opengis.net/waterml/2.0}time').text)
    result['value'] = float(tvp.find('{http://www.opengis.net/waterml/2.0}value').text)
    return result