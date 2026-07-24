default:
    @just --list --unsorted

remove_project PROJECT_NAME:
    git rm --cached -f experiments/projects/{{PROJECT_NAME}}
    rm -rf experiments/projects/{{PROJECT_NAME}}
    rm -rf .git/modules/experiments/projects/{{PROJECT_NAME}}
    git config --remove-section submodule.experiments/projects/{{PROJECT_NAME}}

add_project AUTHOR PROJECT_NAME:
    git submodule add https://github.com/{{AUTHOR}}/{{PROJECT_NAME}} experiments/projects/{{PROJECT_NAME}}
