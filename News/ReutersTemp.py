import os
import pickle
from datetime import timedelta, date, datetime

import bs4
import requests


def get_soup_from_link(link):
    if not link.startswith('http://in.reuters.com'):
        link = 'http://in.reuters.com' + link
    print(link)
    response = requests.get(link)
    assert response.status_code == 200
    return bs4.BeautifulSoup(response.content, 'html.parser')


def date_range(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def run_full():
    today = datetime.now()
    output_dir = 'output_' + today.strftime('%Y-%m-%d-%HH%MM')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Generating the Full dataset in : {}'.format(output_dir))
    start_date = date(2017, 1, 1)
    end_date = today.date()
    iterations = 0
    for single_date in date_range(start_date, end_date):
        output = []
        string_date = single_date.strftime("%Y%m%d")
        link = 'http://in.reuters.com/resources/archive/in/{}.html'.format(string_date)
        try:
            soup = get_soup_from_link(link)
            targets = soup.find_all('div', {'class': 'headlineMed'})
        except Exception:
            print('EXCEPTION RAISED. Could not download link : {}. Resuming anyway.'.format(link))
            targets = []
        for target in targets:
            try:
                timestamp = str(string_date) + str(target.contents[1])
            except Exception:
                timestamp = None
                print('EXCEPTION RAISED. Timestamp set to None. Resuming.')
            title = str(target.contents[0].contents[0])
            if topic in title.lower():
                href = str(target.contents[0].attrs['href'])
                print('iterations = {}, date = {}, ts = {}, t = {}, h= {}'.format(str(iterations).zfill(9), string_date,
                                                                                  timestamp, title, href))
                output.append({'ts': timestamp, 'title': title, 'href': href})
                iterations += 1

        output_filename = os.path.join(output_dir, string_date + '.pkl').format(output_dir, string_date)
        with open(output_filename, 'wb') as w:
            pickle.dump(output, w)
        print('-> written dump to {}'.format(output_filename))


if __name__ == '__main__':
    topic = "infosys"
    run_full()