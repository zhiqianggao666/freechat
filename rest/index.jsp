<%@ page contentType="text/html; charset=GBK" import="java.util.*"%>

<!DOCTYPE html>
<html>
<head>
    <meta charset="charset=gb2312">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
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
    //����XMLHttpRequest
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
                    alert("��Ϣ����Ϊ��");
                    return;
                }
                strI = '<div class="atalk"><span>��:' + TalkWords.value +'</span></div>';
                //get�����ַ���
                var url="http://109.123.123.140:8098/reply?sessionId=1&question="+TalkWords.value;
                Words.innerHTML = Words.innerHTML + strI;
                TalkWords.value="";
                TalkWords.focus();
                Words.scrollTop=Words.scrollHeight;                
                 //���÷�������XMLHttpRequest����
                XmlHttpRequest = createXmlHttpRequest();
                 //���ûص�����
                XmlHttpRequest.onreadystatechange=finish;
                 //��ʼ��xmlhttprequest
                 XmlHttpRequest.open("GET",url,true);
                 //��������
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
                    strBot = '<div class="btalk"><span>������:' + result +'</span></div>' ;  
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
<p align="center"><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;">��ӭ����FreeChat[���ĶԻ�ϵͳ]</span><br /><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;">��ϵͳ���ڿ��������У�лл��</span><br /><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;">��ʵ�֣�TF RNN������ģ��</span><br /><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;"><span style="font-family: NSimSun; font-size: 24px;"> ��ʵ�֣��ں�ʸ��ģ��</span></span></p>
<p align="center"><span style="color: #00ff00; font-family: NSimSun; font-size: 24px;"><span style="font-family: NSimSun; font-size: 24px;">xfei.zhang(henry860916@126.com)</span></span></p>
  <div class="talk_con">
        <div class="talk_show" id="words">
            <div class="atalk"><span id="asay">�ң����</span></div>
            <div class="btalk"><span id="bsay">�����ˣ��ð�</span></div>
        </div>
        <div class="talk_input">
            <input type="text" class="talk_word" id="talkwords">
            <input type="button" value="����" class="talk_sub" id="talksub">
        </div>
    </div>
</body>
</html>