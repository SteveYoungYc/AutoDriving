import urllib
# import urllib2
import requests
import re
import http.cookiejar
from PIL import Image
import time
import json
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np

# 你的header文件
headers = {
    'Host': 'www.95598.cn',
    'User-Agent': '',
    'Accept': '',
    'Accept-Language': '',
    'Accept-Encoding': '',
    'DNT': '1',
    'Referer': '',
    # 'Cookie': ',
    'Connection': 'keep-alive'
}
# 建立一个会话，可以把同一用户的不同请求联系起来；直到会话结束都会自动处理cookies
session = requests.Session()
# 建立LWPCookieJar实例，可以存Set-Cookie3类型的文件。
# 而MozillaCookieJar类是存为'/.txt'格式的文件
session.cookies = http.cookiejar.LWPCookieJar("cookie")
# 若本地有cookie则不用再post数据了
try:
    session.cookies.load(ignore_discard=True)
except IOError:
    print('Cookie未加载！')


def get_code():
    """
    获取验证码本地显示
    返回你输入的验证码，目前还没能自动识别，在更新
    """
    t = str(int(time.time() * 1000))  # 验证码路径是Unix的时间戳作为路径参数
    captcha_url = 'http://www.95598.cn/95598/imageCode/getImgCode?' + t
    response = session.get(captcha_url, headers=headers)
    with open('cptcha.gif', 'wb') as f:
        f.write(response.content)
    # Pillow显示验证码
    #   m = Image.open('cptcha.gif')
    #   m.show()
    # return m
    lena = mpimg.imread('cptcha.gif')  # 读取和代码处于同一目录下的 lena.png
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
    lena.shape  # (512, 512, 3)
    plt.imshow(lena)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    captcha = input('本次登录需要输入验证码： ')
    return captcha


# code = get_code()
# print (code)
def login():
    """
    输入自己的账号密码，模拟登录
    """
    url = 'http://www.95598.cn/95598/userlogin/login'
    data = {
        'url': '/95598/per/account/initSmartConsInfo?partNo=PM02001007',
        'loginway': '01',
        'loginName': '用户名',
        'pwd': '密码',
        'txPwd': '请输入密码',
        'code': get_code()
    }
    result = session.post(url, data=data, headers=headers,
                          allow_redirects=False)  # allow_redirects=False是禁止登陆成功的重定向，返回登陆成功的cookie

    # 开始为了重定向测试用的
    # location = result.headers['Location']
    # location = ""
    # r = session.get(location, cookies = result.cookies, headers = headers, allow_redirects=True)
    # print((json.loads(result.text))['msg'])

    # 保存cookie到本地

    session.cookies.save(ignore_discard=True, ignore_expires=True)

    #     request = urllib.Request(firstURL)
    #     response = urllib.urlopen(request)
    #     content = response.read()
    #     print (content)
    #     r = requests.get(firstURL, headers=headers, timeout=30)
    # r.raise_for_status()
    # r.encoding = r.apparent_encoding
    return result.headers, result.status_code, result.headers['Location']


def isLogin():
    # 通过查看用户个人信息来判断是否已经登录
    url = "查询电费页面URL"
    # 禁止重定向，否则登录失败重定向到首页也是响应200
    login_code = session.get(url, headers=headers, allow_redirects=False).status_code
    if login_code == 200:
        #         r = session.get(url, headers=headers, allow_redirects=False)
        #         print (r.text)
        return True
    else:
        return False


# json数据请求，登陆成功请求json数据就好，不用请求html，还得解析很费劲
def isjson():
    fee_url = "http://www.95598.cn/95598/per/account/getAccountyyt?partNo=PM02001001"
    # 请求json的头文件
    headersfee = {
        'Host': 'www.95598.cn',
        'User-Agent': '',
        'Accept': '',
        'Accept-Language': '',
        'Accept-Encoding': '',
        'DNT': '',
        'X-Requested-With': '',
        'Referer': '',
        # 'Cookie': '',
        'Connection': 'keep-alive',
        'Cache-Control': '',
        'Content-Length': ''
    }
    data_fee = {
        'partNo': 'PM02001001'
    }
    fee_r = session.post(fee_url, data_fee, headersfee, False)
    fee_json = json.loads(fee_r.text)
    # print (fee_json)
    return fee_json


# 解析json，提取余额
def rejson(feejson):
    try:
        z = str(feejson['purchaseInfo']['accountBal'])
        print(z + '\n' + "查询成功!")
        return True
    except Exception as e:
        print('网络错误查询失败，请重试！')


if __name__ == '__main__':
    if isLogin():
        print('您已经登录')
        m = isjson()
        print(m)
        rejson(m)
    else:
        a, b, c = login()
        print(a, b, c)
        if isLogin():
            print('您已经登录')
            n = isjson()
            print(n)
            rejson(n)
        else:
            print('登录失败')
