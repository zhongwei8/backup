import click


@click.command()
@click.option('--dir')  # 可选参数，默认为 None
@click.argument('file-path')  # 必需参数
@click.option('--sum/--no-sum', default=False)  # 开关
@click.option('--shout', is_flag=True)  # 双划线，布尔参数
@click.option('-c', '--check', is_flag=True)  # 单划线，布尔参数
@click.option('-s', '--string')
def info(file_path, dir, sum, shout, check, string):
    click.echo(f'file_path, dir = {file_path}, {dir}')
    click.echo(f'sum = {sum}')
    click.echo(f'shout = {shout}')
    click.echo(f'check = {check}')
    click.echo(f'string = {string}')


if __name__ == "__main__":
    info()
