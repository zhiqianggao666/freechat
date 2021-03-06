<%@ page contentType="text/html; charset=utf-8" import="java.util.*"%>

<!DOCTYPE html>
<html>
<head>
    <meta charset="charset=utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>AI----人机对话</title>
    <style type="text/css">
        .talk_con{
            width:600px;
            height:500px;
            border:1.5px solid  #BC8F8F;
            margin:50px auto 0;
            background:rgba(0,0,0,0);
        }
        .talk_show{
            width:580px;
            height:420px;
            border:1.5px solid  #BC8F8F;
            background:rgba(0,0,0,0);
            margin:10px auto 0;
            overflow:auto;
        }
        .talk_input{
            width:580px;
            margin:10px auto 0;
        }
        .whotalk{
            width:80px;
            height:30px;
            float:left;
            outline:none;
        }
        .talk_word{
            width:420px;
            height:26px;
            padding:0px;
            float:left;
            margin-left:10px;
            outline:none;
            text-indent:10px;
        }        
        .talk_sub{
            width:56px;
            height:30px;
            float:left;
            margin-left:10px;
        }
        .atalk{
           margin:10px; 
        }
        .atalk span{
            display:inline-block;
            background:#0181cc;
            border-radius:10px;
            color:#fff;
            padding:5px 10px;
        }
        .btalk{
            margin:10px;
            text-align:right;
        }
        .btalk span{
            display:inline-block;
            background:#ef8201;
            border-radius:10px;
            color:#fff;
            padding:5px 10px;
        }
    </style>

    <script type="text/javascript">     
    //创建XMLHttpRequest
    function createXmlHttpRequest(){
        if(window.XMLHttpRequest){
            return new XMLHttpRequest();
        }else{
            return new ActiveXObject("Microsoft.XMLHTTP");
        }
    }
        window.onload = function(){
            var Words = document.getElementById("words");
            var TalkWords = document.getElementById("talkwords");
            var TalkSub = document.getElementById("talksub");

            TalkSub.onclick = function(){
                var str = "";
                if(TalkWords.value == ""){
                    alert("消息不能为空");
                    return;
                }
                strI = '<div class="atalk"><span>我:' + TalkWords.value +'</span></div>';
                //get请求字符串
                var url="http://127.0.0.1:8098/reply?sessionId=1&question="+TalkWords.value;
                Words.innerHTML = Words.innerHTML + strI;
                TalkWords.value="";
                TalkWords.focus();
                Words.scrollTop=Words.scrollHeight;                
                 //调用方法创建XMLHttpRequest对象
                XmlHttpRequest = createXmlHttpRequest();
                 //设置回调函数
                XmlHttpRequest.onreadystatechange=finish;
                 //初始化xmlhttprequest
                 XmlHttpRequest.open("GET",window.encodeURI(url),true);
                 //发送请求
                XmlHttpRequest.send(null);
            }
            TalkWords.onkeypress= function EnterPress(e){
            	var e = e || window.event;
            	if(e.keyCode == 13){
            		TalkSub.onclick()
            	}
            }
            function finish(){
                if(XmlHttpRequest.readyState == 4 && XmlHttpRequest.status == 200){
                    var result = XmlHttpRequest.responseText;
                    strBot = '<div class="btalk"><span>机器人:' + result +'</span></div>' ;  
                    Words.innerHTML = Words.innerHTML + strBot;
                    TalkWords.value="";
                    TalkWords.focus();
                    Words.scrollTop=Words.scrollHeight;
                }
            }            

        }


    </script>
</head>
<body style="background-image:url(timg.jpg);background-size:100% auto ; background-repeat:no-repeat ;-moz-background-size:100% 100%;">
<p>
<p align="center" style="color: #000000; text-transform: none; text-indent: 0px; letter-spacing: normal; font-family: 'Times New Roman'; font-size: medium; font-style: normal; font-weight: 400; word-spacing: 0px; white-space: normal; orphans: 2; widows: 2; font-variant-ligatures: normal; font-variant-caps: normal; -webkit-text-stroke-width: 0px; text-decoration-style: initial; text-decoration-color: initial;"><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;">欢迎来到FreeChat[中文对话（骂人）系统]</span><br /><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;">本系统正在开发完善中，谢谢！</span><br /><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;">已实现：TF RNN神经网络模型</span><br /><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;"><span style="font-family: NSimSun; font-size: 24px;">待实现：融合矢量模型,替换免费语料库（目前语料库偏向于骂人的对话）</span></span></p>
<p align="center" style="color: #000000; text-transform: none; text-indent: 0px; letter-spacing: normal; font-family: 'Times New Roman'; font-size: medium; font-style: normal; font-weight: 400; word-spacing: 0px; white-space: normal; orphans: 2; widows: 2; font-variant-ligatures: normal; font-variant-caps: normal; -webkit-text-stroke-width: 0px; text-decoration-style: initial; text-decoration-color: initial;"><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;"><span style="font-family: NSimSun; font-size: 24px;">xfei.zhang(henry860916@126.com)</span></span></p> 
  <div class="talk_con">
        <div class="talk_show" id="words">
            <div class="atalk"><span id="asay">我：你好</span></div>
            <div class="btalk"><span id="bsay">机器人：你 好 啊 呵 呵 呵 呵！</span></div>
        </div>
        <div class="talk_input">
            <input type="text" class="talk_word" id="talkwords">
            <input type="button" value="发送" class="talk_sub" id="talksub">
        </div>
    </div>
</body>
</html>