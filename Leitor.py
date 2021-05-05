import cv2
import numpy as np
import utlis
import time


########################################################################
pathImage = "imagens/2.jpg"
img = cv2.imread(pathImage)
heightImg = 700
widthImg  = 700
questions=5
choices=5
ans= [1,2,0,2,4]
########################################################################

while True:

    img = cv2.resize(img, (widthImg, heightImg)) # Redimensiona a imagem.
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # Cria uma imagem em branco para teste de debug, caso necessário.
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #  Converte a imagem para tons de cinza.
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # Adiciona desfoque.
    imgCanny = cv2.Canny(imgBlur,10,70) # Aplica canny.

    try:
        ## Busca todos os contornos.
        imgContours = img.copy() # Copia a imagem para exibição.
        imgBigContour = img.copy() # Copia a imagem para exibição.
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Busca todos os contornos.
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) #  Desenha todos os contornos encontrados.
        rectCon = utlis.rectContour(contours) # Busca por contornos retangulares.
        biggestPoints= utlis.getCornerPoints(rectCon[0]) # Busca os cantos do retângulo maior.
        gradePoints = utlis.getCornerPoints(rectCon[1]) # Busca os cantos do retângulo menor.

        if biggestPoints.size != 0 and gradePoints.size != 0:

            #  Deforma o maior retângulo.
            biggestPoints=utlis.reorder(biggestPoints) # Reorganiza para a deformação.
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20) # Desenha o maior contorno.
            pts1 = np.float32(biggestPoints) # Prepara os pontos para deformação.
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # Prepara os pontos para deformação.
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # Pega a matriz de transformação.
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) # Aplica a deformação.

            # # Deforma o menor retângulo
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20) # Desenha o maior contorno.
            gradePoints = utlis.reorder(gradePoints) #  Reorganiza para a deformação.
            ptsG1 = np.float32(gradePoints)  # Prepara os pontos para deformação.
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # Prepara os pontos para deformação.
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2) # Pega a matriz de transformação.
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150)) # Aplica a deformação.

            # Aplica a borda.
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY) # Converte para tons de cinza.
            imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # Aplica a borda e a inverte.

            boxes = utlis.splitBoxes(imgThresh) # Pega os espaços individuais.
            # cv2.imshow("Split Test ", boxes[3])
            countR=0
            countC=0
            myPixelVal = np.zeros((questions,choices)) # Guarda os valores não nulos em cada espaço.
            for image in boxes:
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC]= totalPixels
                countC += 1
                if (countC==choices):countC=0;countR +=1
                
            # Busca as respostas do usuário e as coloca em uma lista.
            myIndex=[]
            for x in range (0,questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print("USER ANSWERS",myIndex)

            # Compara os valores para encontrar os valores corretos.
            grading=[]
            for x in range(0,questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:grading.append(0)
            print("GRADING",grading)
            score = (sum(grading)/questions)*100 # Média final.
            print("SCORE",score)

            # Exibe as respostas.
            utlis.showAnswers(imgWarpColored,myIndex,grading,ans) # Desenha as respostas detectadas.
            utlis.drawGrid(imgWarpColored) # Desenha a malha.
            imgRawDrawings = np.zeros_like(imgWarpColored) # Nova imagem em branco com tamanho da imagem distorcida.
            utlis.showAnswers(imgRawDrawings, myIndex, grading, ans) #  Desenha na nova imagem.
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1) # Inverte a matriz de transformação.
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg)) 

            # Exibe a média.
            imgRawGrade = np.zeros_like(imgGradeDisplay,np.uint8) # Adiciona nova imagem em branco com espaço para média.
            cv2.putText(imgRawGrade,str(int(score))+"%",(50,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),2) # Adiciona a média à nova imagem,.
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1) # Inverte a matriz de transformação.
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg)) 

            # Mostra as respostas e a média na imagem final.
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0) # Aciona as bolinhas de resposta correta.
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0) # Adiciona o aproveitamento.
            cv2.imshow("teste", imgRawGrade)

            # Vetor da imagem para exibição.
            imageArray = ([img,imgGray,imgCanny,imgContours],
                            [imgBigContour,imgThresh,imgWarpColored,imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours],
                        [imgBlank, imgBlank, imgBlank, imgBlank])

    # Rótulos para exibição.
    lables = [["Original","Cinza","Limites","Contornos"],
                ["Vertices","Negativo","Matriz","Final"]]

    stackedImage = utlis.stackImages(imageArray,0.5,lables)
    cv2.imshow("Result",stackedImage)

    # Salva quando a tecla 's' é pressionada.
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("./gabarito.jpg",imgFinal)
        cv2.imwrite("./gabarito_aproveitamento.jpg",imgRawGrade)
       
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                        (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Gabarito Salvo!", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        break