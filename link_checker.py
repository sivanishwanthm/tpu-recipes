
import os
import re
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_files(directories, extensions):
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    yield os.path.join(root, file)

def extract_links(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        return []

    # Regex for Markdown and HTML links
    links = re.findall(r'\[.*?\]\((?!#)(.*?)\)|<a\s+(?:[^>]*?\s+)?href="(?!#)([^"]*)"', content)

    # Flatten the list of tuples, remove empty strings, and filter out anchor links
    return [item for sublist in links for item in sublist if item and not item.startswith('#')]

def check_link(link, base_dir):
    parsed_url = urlparse(link)

    if parsed_url.scheme in ['http', 'https']:
        # External link
        for attempt in range(3):
            try:
                response = requests.head(link, allow_redirects=True, timeout=30)
                if response.status_code < 400:
                    return None, None
                else:
                    return link, response.status_code
            except requests.RequestException as e:
                if attempt == 2:
                    return link, str(e)
    else:
        # Internal link
        # Ignore mailto links
        if parsed_url.scheme == 'mailto':
            return None, None

        # Join the base directory with the link, then normalize the path
        path = os.path.normpath(os.path.join(base_dir, parsed_url.path))

        # If the link is a directory, check for README.md or index.html
        if os.path.isdir(path):
            if not any(os.path.exists(os.path.join(path, index)) for index in ['README.md', 'index.html']):
                 return link, "Points to a directory with no index file."
        elif not os.path.exists(path):
            return link, '404 Not Found'

    return None, None

def generate_report(broken_links):
    """Generates a markdown report of broken links."""
    with open("broken_links_report.md", "w", encoding="utf-8") as f:
        if not broken_links:
            f.write("# Link Check Report\n\n")
            f.write("No broken links found.\n")
            return

        f.write("# Broken Link Report\n\n")
        f.write("| File Path | Broken URL | HTTP Error Code |\n")
        f.write("|---|---|---|\n")
        for file, link, status in sorted(broken_links):
            f.write(f"| {file} | {link} | {status} |\n")

def main():
    files_to_check = list(find_files(['.'], ['.md', '.MDX', '.html']))
    broken_links = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_link = {}
        for file in files_to_check:
            base_dir = os.path.dirname(file)
            links = extract_links(file)
            for link in links:
                future = executor.submit(check_link, link, base_dir)
                future_to_link[future] = (file, link)

        for future in as_completed(future_to_link):
            file, link = future_to_link[future]
            broken_link, status = future.result()
            if broken_link:
                broken_links.append((file, broken_link, status))

    generate_report(broken_links)
    if broken_links:
        print(f"Found {len(broken_links)} broken links. Report generated at broken_links_report.md")
    else:
        print("No broken links found. Report generated at broken_links_report.md")

if __name__ == "__main__":
    main()
