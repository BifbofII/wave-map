import pandas as pd
import pandas_read_xml as pdx

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