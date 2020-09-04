import pandas as pd
import pandas_read_xml as pdx
import xml.etree.ElementTree as ET

def_server = 'https://opendata.fmi.fi'
def_base_query = '/wfs?service=WFS&version=2.0.0&'

url = 'https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=describeStoredQueries&'

class FmiDownloader:
    def __init__(self, server=def_server, base_query=def_base_query):
        self.server = server
        self.base_query = base_query
        self.stored_queries = self._download_stored_queries()

    def get_stored_query_list(self):
        return self.stored_queries

    def _download_stored_queries(self):
        url = self._build_query_url({'request': 'describeStoredQueries'})
        xml = pdx.read_xml_from_url(url)
        df = pdx.read_xml(xml, ['DescribeStoredQueriesResponse', 'StoredQueryDescription'])
        df = df.set_index('@id')
        return df

    def _build_query_url(self, parameters):
        url = self.server + self.base_query
        for key, val in parameters.items():
            url += key + '=' + val + '&'
        return url

    def find_queries(self, search_term, search_in=['Title', 'Abstract'], lower=True):
        mask = pd.Series(False, index=self.stored_queries.index)
        for c in search_in:
            if lower:
                cont = self.stored_queries[c].str.lower()
            else:
                cont = self.stored_queries[c]
            mask = mask | cont.str.contains(search_term)
        return self.stored_queries[mask]

    def download_data(self, stored_query_id, **params):
        # Download XML data
        params.update({'request': 'getFeature', 'storedquery_id': stored_query_id})
        url = self._build_query_url(params)
        xml = pdx.read_xml_from_url(url)
        self._handle_query_error(xml)

        # Find and handle data type
        id_parts = stored_query_id.split('::')
        if id_parts[-1] == 'simple':
            return self._parse_simple_data(xml)
        elif id_parts[-1] == 'timevaluepair':
            return self._parse_tvp_data(xml)
        elif id_parts[-1] == 'multipointcoverage':
            return self._parse_mpc_data(xml)
        elif id_parts[-1] == 'grid':
            return self._parse_grid_data(xml)
        
        raise ValueError('Query of unknown data type')

    @staticmethod
    def _handle_query_error(xml):
        root = ET.fromstring(xml)
        if 'ExceptionReport' not in root.tag:
            return
        for ex in root:
            raise QueryException(ex)

    @staticmethod
    def _parse_simple_data(xml):
        raise NotImplementedError

    @staticmethod
    def _parse_tvp_data(xml):
        raise NotImplementedError

    @staticmethod
    def _parse_mpc_data(xml):
        raise NotImplementedError

    @staticmethod
    def _parse_grid_data(xml):
        raise NotImplementedError

class QueryException(Exception):
    def __init__(self, xml_exception):
        self.xml_exception = xml_exception
        self.exception_code = xml_exception.attrib['exceptionCode']
        self.text = ''
        for ex_tx in xml_exception:
            self.text += ex_tx.text + '\n'

    def __str__(self):
        return '{}: {}'.format(self.exception_code, self.text)