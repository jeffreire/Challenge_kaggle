nome = input('Informe seu nome: ')
idade = int(input('Informe sua idade: '))

nome = nome.lower()
if nome == 'marcelo':
    print('Bem vindo Cabecudo ')
else:
    print('Bem vindo Bunitao')

while True:
    X = input('{} voce se sente bonito?'.format( nome) )
    if X == 'sim':
        print('Mentiroso')
        input()
        break
    elif X == 'nao':
        print('Voce é bonito porque voce nao mentiu')
        input()
        break
    else:
        print('responda direito')
        True 