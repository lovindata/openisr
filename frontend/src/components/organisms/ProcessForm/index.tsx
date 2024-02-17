import { components } from "../../../services/backend/endpoints";
import { BorderBox } from "../../atoms/BorderBox";
import { Header } from "../../atoms/Header";
import { ConfigurationContents } from "./ConfigurationContents";
import { ErrorContents } from "./ErrorContents";
import { Tab } from "@headlessui/react";

interface Props {
  image: components["schemas"]["ImageODto"];
  latestProcess?: components["schemas"]["ProcessODto"];
  onSuccessSubmit?: () => void;
}

export function ProcessForm({ image, latestProcess, onSuccessSubmit }: Props) {
  return (
    <BorderBox className="w-72 space-y-3 bg-black p-4">
      <Tab.Group>
        <Tab.List className="space-x-3">
          {latestProcess?.status.ended?.kind === "failed" && (
            <>
              <Tab className="outline-none">
                {({ selected }) => (
                  <Header
                    name="Error"
                    className={selected ? "select-none" : "opacity-50"}
                  />
                )}
              </Tab>
              <span className="border-x" />
            </>
          )}
          <Tab className="outline-none">
            {({ selected }) => (
              <Header
                name="Configurations"
                className={selected ? "select-none" : "opacity-50"}
              />
            )}
          </Tab>
        </Tab.List>
        <Tab.Panels>
          {latestProcess?.status.ended?.kind === "failed" && (
            <Tab.Panel>
              <ErrorContents
                error={latestProcess.status.ended.error}
                image={image}
                onSuccessSubmit={onSuccessSubmit}
              />
            </Tab.Panel>
          )}
          <Tab.Panel>
            <ConfigurationContents
              image={image}
              latestProcess={latestProcess}
              onSuccessSubmit={onSuccessSubmit}
            />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </BorderBox>
  );
}
