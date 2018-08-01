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
            border:1px solid #666;
            margin:50px auto 0;
            background:#f9f9f9;
        }
        .talk_show{
            width:580px;
            height:420px;
            border:1px solid #666;
            background:#fff;
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
            //var Who = document.getElementById("who");
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
<body>
��ӭ����FreeChat[���ĶԻ�ϵͳ]��<br>
��ϵͳ���ڿ��������У�лл��<br>
��ʵ�֣�TF RNN������ģ��<br>
��ʵ�֣��ں�ʸ��ģ��
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