# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:36:09 2021

@author: vegex
"""

#!/usr/bin/env python
print('If you get error "ImportError: No module named \'six\'" install six:\n'+\
'$ sudo pip install six');
print('To enable your free eval account and get CUSTOMER, YOURZONE and ' + \
'YOURPASS, please contact sales@luminati.io')
import sys
if sys.version_info[0]==2:
    import six
    from six.moves.urllib import request
    import random
    username = 'lum-customer-c_fb7e1152-zone-static-route_err-block'
    password = '5abk7h9ihbd5'
    port = 22225
    session_id = random.random()
    super_proxy_url = ('http://%s-session-%s:%s@zproxy.lum-superproxy.io:%d' % (username, session_id, password, port))
    proxy_handler = request.ProxyHandler({
            'http': super_proxy_url,
            'https': super_proxy_url,
            })
    opener = request.build_opener(proxy_handler)
    opener.addheaders = \
        [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36')]
    print('Performing request')
        
    print(opener.open('http://lumtest.com').read())

if sys.version_info[0]==3:
    import urllib.request
    import random
    username = 'lum-customer-c_fb7e1152-zone-static-route_err-block'
    password = '5abk7h9ihbd5'
    port = 22225
    session_id = random.random()
    super_proxy_url = ('http://%s-session-%s:%s@zproxy.lum-superproxy.io:%d' %
    (username, session_id, password, port))
    proxy_handler = urllib.request.ProxyHandler({
    'http': super_proxy_url,
    'https': super_proxy_url,
    })
    opener = urllib.request.build_opener(proxy_handler)
    opener.addheaders = \
    [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36')]
    print('Performing request')
    print(opener.open('http://lumtest.com'))