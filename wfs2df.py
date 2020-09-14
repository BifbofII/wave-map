import os.path
import dateutil.parser
import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as et


def parse_wfs(xml_file, station_xml=None):
    """
    Parse XML data in GML format of a WFS server to a pandas dataframe

    The data is parsed and returned as a pandas dataframe.
    Additionally some of the metadata is returned as a dict.

    Currently only parsing of time value pair data is supported.

    :param xml_file: the path to an xml file or a file object
    :param station_xml: a xml file with information about the stations (optional)
    :returns: pandas dataframe containing the data and a dict containing additional metadata
    :raises ValueError: when invalid xml data is given
    """
    if type(xml_file) != str:
        cursor_pos = xml_file.tell()
    tree = et.parse(xml_file)
    if type(xml_file) != str:
        xml_file.seek(cursor_pos)
    xml_root = tree.getroot()

    if xml_root.tag != '{http://www.opengis.net/wfs/2.0}FeatureCollection':
        raise ValueError('Invalid GML data')

    metadata = xml_root.attrib
    del metadata['{http://www.w3.org/2001/XMLSchema-instance}schemaLocation']

    metadata['gpd parse'] = gpd.read_file(xml_file)
    if type(xml_file) != str:
        xml_file.seek(cursor_pos)

    if station_xml:
        all_stations, _ = parse_wfs(station_xml)
        included_ids = np.unique(metadata['gpd parse'].identifier)
        metadata['stations'] = all_stations.loc[included_ids,:]

    data = []
    for member in xml_root.iterfind('{http://www.opengis.net/wfs/2.0}member'):
        data.append(_parse_member(member))

    data = pd.concat(data)
    if len(data.columns[0].split('-')) > 1:
        index = pd.MultiIndex.from_tuples(zip(metadata['gpd parse'].identifier, map(lambda i: i.split('-')[-1], data.columns)))
        data.columns = index
    
    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='all')

    return data, metadata


def _parse_member(member):
    """
    Parse a WFS member element

    :param member: a member xml element
    :returns: a pandas dataframe with the data of the member
    """
    data = []
    for ts in member.iterfind('{http://inspire.ec.europa.eu/schemas/omso/3.0}PointTimeSeriesObservation'):
        data.append(_parse_ts(ts))
    for emf in member.iterfind('{http://inspire.ec.europa.eu/schemas/ef/4.0}EnvironmentalMonitoringFacility'):
        data.append(_parse_facility(emf))

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


def _parse_facility(emf):
    """
    Parse a environmental monitoring facility entry

    :param emf: the environmental monitoring facility xml element
    :returns: a pandas dataframe with the information about the emf
    """
    data = dict()
    data['id'] = int(emf.find('{http://www.opengis.net/gml/3.2}identifier').text)
    for n in emf.iterfind('{http://www.opengis.net/gml/3.2}name'):
        code = n.attrib['codeSpace'].split('/')[-1]
        try:
            data[code] = int(n.text)
        except ValueError:
            data[code] = n.text
    data['pos'] = _parse_point(emf.find('{http://inspire.ec.europa.eu/schemas/ef/4.0}representativePoint/{http://www.opengis.net/gml/3.2}Point'))
    data = pd.DataFrame([data])
    data = data.set_index('id')
    return data


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

    :param tvp: a TVP xml element
    :returns: a dict with time and value entries
    """
    result = dict()
    result['time'] = dateutil.parser.isoparse(tvp.find('{http://www.opengis.net/waterml/2.0}time').text)
    result['value'] = float(tvp.find('{http://www.opengis.net/waterml/2.0}value').text)
    return result

def _parse_point(point):
    """
    Parse a GML point

    :param point: a point xml element
    :returns: a tuple with (long, lat)
    """
    xy = point.find('{http://www.opengis.net/gml/3.2}pos').text.split()
    return (float(xy[0]), float(xy[1]))