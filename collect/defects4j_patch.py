import os
import shutil

path_defects4j = '/Users/haoye.tian/Documents/University/project/defects4j/framework/projects/'
defects4j_patch = '/Users/haoye.tian/Documents/University/project/defects4j_patch/'
projects = ['Chart', 'Cli', 'Closure', 'Codec', 'Collections', 'Compress', 'Csv', 'Gson', 'JacksonCore', 'JacksonDatabind', 'JacksonXml', 'Jsoup', 'JxPath', 'Lang', 'Math', 'Mockito', 'Time']

cnt = 0
success = 0
for project in projects:
    for root, dirs, files in os.walk(path_defects4j+project+ '/patches'):
        for file in files:
            if file.endswith('.src.patch'):
                cnt += 1
                id = file.split('.')[0]
                patch = ''
                try:
                    with open(os.path.join(root, file), 'r+', newline='\r\n') as f:
                        for line in f:
                            if line.startswith('-') and not line.startswith('---'):
                                patch += '+' + line[1:]
                            elif line.startswith('+') and not line.startswith('+++'):
                                patch += '-' + line[1:]
                            else:
                                patch += line
                except Exception as e:
                    print('name:{}'.format(project+' '+id))

                # write into patch
                new_path = defects4j_patch + project
                new_name = id+'.patch'
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                with open(os.path.join(new_path, new_name), 'w+') as f:
                    f.write(patch)
                success += 1

                print('success/cnt: {}/{}'.format(success, cnt))
